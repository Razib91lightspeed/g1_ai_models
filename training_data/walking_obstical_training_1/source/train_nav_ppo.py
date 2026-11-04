#!/usr/bin/env python3
"""Navigation command-policy PPO in a 2-D kinematic proxy; NOT Isaac Lab G1 dynamics."""
import argparse, csv, json, math, os, random, signal, sys, time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from torch import nn
from torch.distributions import Categorical

STOP = False

def request_stop(signum, frame):
    global STOP
    STOP = True

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, request_stop)

SPEEDS = (0.0, 0.12, 0.24, 0.38)
YAWS = (-0.8, -0.4, 0.0, 0.4, 0.8)
N_ACTIONS = len(SPEEDS) * len(YAWS)
N_RAYS = 12
OBS_DIM = 3 + N_RAYS + 2
DT = 0.10
TZ = ZoneInfo("Europe/Helsinki")

class NavEnv:
    def __init__(self, n, device):
        self.n, self.device = n, device
        self.pos = torch.zeros(n, 2, device=device)
        self.goal = torch.zeros(n, 2, device=device)
        self.heading = torch.zeros(n, device=device)
        self.speed = torch.zeros(n, device=device)
        self.yaw = torch.zeros(n, device=device)
        self.steps = torch.zeros(n, device=device, dtype=torch.long)
        self.path = torch.zeros(n, device=device)
        self.init_dist = torch.zeros(n, device=device)
        self.obs_xy = torch.zeros(n, 3, 2, device=device)
        self.obs_rad = torch.zeros(n, 3, device=device)
        self.obs_vel = torch.zeros(n, 3, 2, device=device)
        self.active_obs = 0
        self.reset(torch.arange(n, device=device), stage=0)

    @torch.no_grad()
    def reset(self, ids, stage):
        k = ids.numel()
        if not k:
            return
        d = self.device
        start = (torch.rand(k, 2, device=d) - .5) * 2.
        angle = (torch.rand(k, device=d) * 2 - 1) * math.pi
        dist = 2.0 + torch.rand(k, device=d) * (3.0 if stage == 0 else 5.0)
        line = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        side = torch.stack((-line[:, 1], line[:, 0]), -1)
        self.pos[ids] = start
        self.goal[ids] = start + dist[:, None] * line
        self.heading[ids] = (torch.rand(k, device=d) * 2 - 1) * math.pi
        self.speed[ids] = 0
        self.yaw[ids] = 0
        self.steps[ids] = 0
        self.path[ids] = 0
        self.init_dist[ids] = dist
        self.active_obs = 0 if stage < 2 else 3
        # Obstacles near straight-line route, so agent must learn detours.
        frac = .20 + .60 * torch.rand(k, 3, 1, device=d)
        lateral = (torch.rand(k, 3, 1, device=d) - .5) * 1.8
        self.obs_xy[ids] = start[:, None, :] + dist[:, None, None] * frac * line[:, None, :] + lateral * side[:, None, :]
        self.obs_rad[ids] = .30 + .20 * torch.rand(k, 3, device=d)
        directions = torch.rand(k, 3, device=d) * 2 * math.pi
        self.obs_vel[ids] = torch.stack((directions.cos(), directions.sin()), -1) * .12

    @torch.no_grad()
    def observe(self):
        delta = self.goal - self.pos
        dist = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-5)
        angle = torch.atan2(delta[:, 1], delta[:, 0]) - self.heading
        # Ray-circle intersection for a small egocentric range sensor.
        rays = self.heading[:, None] + torch.arange(N_RAYS, device=self.device)[None, :] * (2 * math.pi / N_RAYS)
        dirs = torch.stack((rays.cos(), rays.sin()), -1)
        if self.active_obs:
            rel = self.obs_xy - self.pos[:, None, :]
            projection = torch.einsum('nrc,noc->nro', dirs, rel)
            center_dist2 = (rel.square().sum(-1))[:, None, :]
            rr = (self.obs_rad + .28)[:, None, :]
            disc = rr.square() - (center_dist2 - projection.square())
            hit = (disc >= 0) & (projection > 0)
            first = projection - torch.sqrt(disc.clamp_min(0))
            length = torch.where(hit, first.clamp_min(0), 3.0).amin(-1)
            lidar = (length / 3.0).clamp(0, 1)
        else:
            lidar = torch.ones(self.n, N_RAYS, device=self.device)
        result = torch.cat((
            torch.stack((dist.clamp(max=10) / 10, angle.sin(), angle.cos()), -1),
            lidar,
            self.speed[:, None] / .38,
            self.yaw[:, None] / .8,
        ), -1)
        return result

    @torch.no_grad()
    def step(self, action, stage):
        old_dist = (self.goal - self.pos).norm(dim=-1)
        target_speed = torch.tensor(SPEEDS, device=self.device)[action // len(YAWS)]
        target_yaw = torch.tensor(YAWS, device=self.device)[action % len(YAWS)]
        # Limit acceleration and yaw command changes.
        previous_speed, previous_yaw = self.speed.clone(), self.yaw.clone()
        self.speed += (target_speed - self.speed).clamp(-.05, .05)
        self.yaw += (target_yaw - self.yaw).clamp(-.2, .2)
        self.heading = torch.remainder(self.heading + self.yaw * DT + math.pi, 2 * math.pi) - math.pi
        displacement = self.speed[:, None] * DT * torch.stack((self.heading.cos(), self.heading.sin()), -1)
        self.pos += displacement
        self.path += displacement.norm(dim=-1)
        self.steps += 1
        if stage >= 3:
            self.obs_xy += self.obs_vel * DT
        new_dist = (self.goal - self.pos).norm(dim=-1)
        collision = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        closest = torch.full((self.n,), 3.0, device=self.device)
        if stage >= 2:
            clearance = (self.obs_xy - self.pos[:, None]).norm(dim=-1) - self.obs_rad - .28
            closest = clearance.amin(-1)
            collision = closest < 0.0
        success = new_dist < .35
        timeout = self.steps >= 300
        done = collision | success | timeout
        # Approximate reward shaping; not a guarantee of globally shortest route.
        reward = (old_dist - new_dist) * 8.0 - .008
        reward -= .018 * (self.speed - previous_speed).abs()
        reward -= .012 * (self.yaw - previous_yaw).abs()
        if stage >= 2:
            reward -= .015 * (1 - closest.clamp(0, 1))
        reward += success.float() * 18.0 - collision.float() * 12.0
        info = {
            'success': success[done].float().cpu().tolist(),
            'collision': collision[done].float().cpu().tolist(),
            'efficiency': (self.init_dist[done] / self.path[done].clamp_min(.1)).clamp(max=1).cpu().tolist(),
            'steps': self.steps[done].cpu().tolist(),
        }
        idx = done.nonzero().flatten()
        self.reset(idx, stage)
        return self.observe(), reward, done, info

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)
    def forward(self, obs):
        hidden = self.net(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

def save_checkpoint(path, agent, optimizer, iteration, stage, totals):
    temp = path.with_suffix('.partial')
    torch.save({'model': agent.state_dict(), 'optimizer': optimizer.state_dict(),
                'iteration': iteration, 'stage': stage, 'totals': totals,
                'observation_dim': OBS_DIM, 'action_speeds': SPEEDS, 'action_yaws': YAWS,
                'model_type': '2D kinematic navigation PPO; not Isaac Lab G1'}, temp)
    os.replace(temp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--max-updates', type=int, default=0)
    p.add_argument('--deadline', required=True, help='YYYY-MM-DDTHH:MM Europe/Helsinki')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-cpu-smoke', action='store_true', help='Local validation only; never use for overnight runs')
    args = p.parse_args()
    if args.envs < 1: p.error('envs must be positive')
    deadline = datetime.fromisoformat(args.deadline).replace(tzinfo=TZ)
    if deadline <= datetime.now(TZ): p.error('deadline is already past')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not (args.smoke and args.allow_cpu_smoke):
        raise RuntimeError('STOP: overnight training requires a visible CUDA GPU; refusing CPU fallback')
    print(f'START mode={"smoke" if args.smoke else "overnight"} device={device} envs={args.envs} deadline={deadline.isoformat()}', flush=True)
    env = NavEnv(args.envs, device)
    agent = Agent().to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    iteration = 0
    totals = {'episodes': 0, 'successes': 0, 'collisions': 0}
    recent_success, recent_efficiency = deque(maxlen=250), deque(maxlen=250)
    metrics = out / 'metrics.csv'
    if not metrics.exists():
        with metrics.open('w', newline='') as f:
            csv.writer(f).writerow(['time_helsinki','iteration','stage','episodes','success_rate_250','path_efficiency_250','total_successes','total_collisions','mean_reward'])
    last_save = time.monotonic()
    start = time.monotonic()
    try:
        while not STOP and datetime.now(TZ) < deadline:
            if args.max_updates and iteration >= args.max_updates: break
            # Fixed curriculum over elapsed wall-clock time, for overnight training.
            frac = min(1.0, (time.monotonic() - start) / max((deadline - datetime.now(TZ)).total_seconds() + time.monotonic() - start, 1.0))
            stage = 0 if frac < .15 else (1 if frac < .35 else (2 if frac < .70 else 3))
            env.active_obs = 0 if stage < 2 else 3
            obs = env.observe()
            rb_obs, rb_actions, rb_logp, rb_rewards, rb_dones, rb_values = [], [], [], [], [], []
            episode_count = 0
            for _ in range(128):
                with torch.no_grad():
                    logits, value = agent(obs)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    logp = distribution.log_prob(action)
                next_obs, reward, done, info = env.step(action, stage)
                rb_obs.append(obs); rb_actions.append(action); rb_logp.append(logp)
                rb_rewards.append(reward); rb_dones.append(done.float()); rb_values.append(value)
                episode_count += len(info['success'])
                recent_success.extend(info['success'])
                recent_efficiency.extend(info['efficiency'])
                totals['episodes'] += len(info['success'])
                totals['successes'] += int(sum(info['success']))
                totals['collisions'] += int(sum(info['collision']))
                obs = next_obs
                if STOP or datetime.now(TZ) >= deadline: break
            if not rb_obs: break
            with torch.no_grad(): last_value = agent(obs)[1]
            advantages = []
            gae = torch.zeros(args.envs, device=device)
            for j in reversed(range(len(rb_obs))):
                nonterminal = 1 - rb_dones[j]
                delta = rb_rewards[j] + .99 * last_value * nonterminal - rb_values[j]
                gae = delta + .99 * .95 * nonterminal * gae
                advantages.append(gae)
                last_value = rb_values[j]
            adv = torch.stack(advantages[::-1]).flatten()
            returns = adv + torch.stack(rb_values).flatten()
            batch_obs = torch.stack(rb_obs).reshape(-1, OBS_DIM)
            batch_act = torch.stack(rb_actions).flatten()
            batch_oldlogp = torch.stack(rb_logp).flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            length = batch_obs.shape[0]
            for _ in range(3):
                for idx in torch.randperm(length, device=device).split(1024):
                    logits, values = agent(batch_obs[idx])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(batch_act[idx]) - batch_oldlogp[idx]).exp()
                    policy_loss = -torch.minimum(ratio * adv[idx], ratio.clamp(.8, 1.2) * adv[idx]).mean()
                    value_loss = .5 * (values - returns[idx]).square().mean()
                    loss = policy_loss + .5 * value_loss - .01 * dist.entropy().mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), .5)
                    opt.step()
            iteration += 1
            if iteration % (1 if args.smoke else 10) == 0:
                now = datetime.now(TZ).isoformat()
                sr = sum(recent_success) / len(recent_success) if recent_success else 0
                pe = sum(recent_efficiency) / len(recent_efficiency) if recent_efficiency else 0
                mr = float(torch.stack(rb_rewards).mean().item())
                with metrics.open('a', newline='') as f:
                    csv.writer(f).writerow([now,iteration,stage,totals['episodes'],round(sr,4),round(pe,4),totals['successes'],totals['collisions'],round(mr,5)])
                print(f'UPDATE {iteration} stage={stage} episodes={totals["episodes"]} recent_success={sr:.3f} efficiency={pe:.3f} mean_reward={mr:.4f}', flush=True)
            if iteration % (2 if args.smoke else 30) == 0 or time.monotonic() - last_save > 600:
                save_checkpoint(out / 'latest.pt', agent, opt, iteration, stage, totals)
                last_save = time.monotonic()
    finally:
        save_checkpoint(out / 'latest.pt', agent, opt, iteration, locals().get('stage', 0), totals)
        with (out / 'summary.json').open('w') as f:
            json.dump({'ended_helsinki': datetime.now(TZ).isoformat(), 'updates': iteration,
                       'totals': totals, 'device': str(device), 'model_type': '2D kinematic navigation proxy, not Isaac Lab G1',
                       'stopped_by_signal': STOP, 'deadline_helsinki': deadline.isoformat()}, f, indent=2)
        print('SAVED_FINAL_CHECKPOINT', out / 'latest.pt', 'updates=', iteration, flush=True)

if __name__ == '__main__': main()
