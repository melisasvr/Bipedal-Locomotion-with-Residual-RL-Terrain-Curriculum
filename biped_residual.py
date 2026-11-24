"""
Planar LIPM biped + capture-point base controller + residual policy (ARS)
+ terrain curriculum. Pure NumPy + Matplotlib. Runs on a modest laptop.

Usage:
    python biped_residual.py train --iters 300
    python biped_residual.py eval
    python biped_residual.py animate
    python biped_residual.py all
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
DX = 0.05          # terrain resolution (m)
M = 1600           # terrain samples -> 80 m
LAM = 0.25         # fraction of potential-energy change felt by the CoM

# ----------------------------------------------------------------------------
# Curriculum stages: max slope (deg), max stair height (m), push size (m^2/s^2-ish)
# ----------------------------------------------------------------------------
STAGES = [
    dict(name="flat",   slope=0.0,  stair=0.0,  push=0.0),
    dict(name="slopes", slope=10.0, stair=0.0,  push=0.0),
    dict(name="stairs", slope=0.0,  stair=0.10, push=0.0),
    dict(name="rough",  slope=12.0, stair=0.15, push=0.0),
    dict(name="pushes", slope=12.0, stair=0.15, push=0.5),
]


# ----------------------------------------------------------------------------
# Environment: step-to-step LIPM, vectorized over N = n_pol * n_eps robots.
# One env step = one footstep. Closed form dynamics, so no integrator needed.
# ----------------------------------------------------------------------------
class BipedEnv:
    OBS, ACT = 10, 4

    def __init__(self, n_pol, n_eps, seed=0, max_steps=60):
        self.P, self.E, self.N = n_pol, n_eps, n_pol * n_eps
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.eidx = np.tile(np.arange(n_eps), n_pol)   # env n uses terrain eidx[n]
        self.zc_nom, self.T0 = 0.8, 0.45
        self.w_nom = np.sqrt(G / self.zc_nom)
        self.record, self.log = False, []

    def _tile(self, x):
        return np.tile(x, self.P)

    def _noise(self):
        return self._tile(self.rng.normal(size=self.E))

    def _uniform(self):
        return self._tile(self.rng.random(self.E))

    # ---- terrain ---------------------------------------------------------
    def _build_terrain(self, slope_deg, stair_h, randomize_amp):
        E, rng = self.E, self.rng
        has_s, has_t = slope_deg > 0, stair_h > 0
        use_slope = np.where(has_s & has_t, rng.random(E) < 0.5, has_s)
        use_stair = has_t & ~use_slope
        amp = rng.uniform(0.3, 1.0, (2, E)) if randomize_amp else np.ones((2, E))
        # slopes: random angle per 3 m segment, integrated into a height profile
        seg = 60
        nseg = M // seg + 1
        ang = np.deg2rad(rng.uniform(-1, 1, (E, nseg)) * (slope_deg * amp[0])[:, None])
        slope = np.repeat(np.tan(ang), seg, axis=1)[:, :M]
        Hs = np.cumsum(slope * DX, axis=1)
        # stairs: plateaus of random width, random up/down height jumps
        wid = rng.integers(8, 17, E)
        plat = np.arange(M)[None, :] // wid[:, None]
        inc = rng.uniform(-1, 1, (E, M)) * (stair_h * amp[1])[:, None]
        Ht = np.take_along_axis(np.cumsum(inc, axis=1), plat, axis=1)
        H = np.where(use_slope[:, None], Hs, 0.0) + np.where(use_stair[:, None], Ht, 0.0)
        k = int(2.0 / DX)                       # flat 2 m start
        H = H - H[:, [k]]
        H[:, :k] = 0.0
        return H

    def _height(self, x):
        idx = np.clip((x / DX).astype(int), 0, M - 1)
        return self.H[self.eidx, idx]

    # ---- reset -----------------------------------------------------------
    def reset(self, slope_deg=0.0, stair_h=0.0, push_mag=0.0, vd=0.8, push_prob=0.10,
              push_at=None, push_val=0.0, randomize_amp=True):
        E = self.E
        b = lambda x: np.broadcast_to(np.asarray(x, float), (E,)).copy()
        self.H = self._build_terrain(b(slope_deg), b(stair_h), randomize_amp)
        self.push_mag = self._tile(b(push_mag))
        self.push_prob, self.push_at, self.push_val = push_prob, push_at, push_val
        self.vd = self._tile(b(vd))
        zc = self.zc_nom * self._tile(self.rng.uniform(0.9, 1.1, E))   # model mismatch
        self.w = np.sqrt(G / zc)
        self.zc = zc
        self.xs = np.zeros(self.N)
        self.hs = np.zeros(self.N)
        self.v = self.vd * (1 + 0.05 * self._noise())
        bdes = self.vd * self.T0 / (np.exp(self.w_nom * self.T0) - 1)
        self.r = bdes - self.v / self.w_nom + 0.02 * self._noise()
        self.last_a = np.zeros((self.N, self.ACT))
        self.alive = np.ones(self.N, bool)
        self.t = 0
        self.effort = np.zeros(self.N)
        self.err = np.zeros(self.N)
        self.n_ok = np.zeros(self.N)
        self.log = []
        return self.obs()

    def obs(self):
        hh = np.stack([self._height(self.xs + d) - self.hs for d in (0.35, 0.7, 1.05)], 1)
        xi = self.r + self.v / self.w_nom
        return np.column_stack([self.v - self.vd, self.r, xi, 5 * hh, self.last_a])

    # ---- one footstep ----------------------------------------------------
    def step(self, act):
        a = np.clip(act, -1, 1)
        dL, dT, c, e = 0.10 * a[:, 0], 0.06 * a[:, 1], 0.04 * a[:, 2], 1.0 * a[:, 3]
        T = self.T0 + dT
        w, r, v = self.w, self.r, self.v
        ch, sh = np.cosh(w * T), np.sinh(w * T)
        rT = (r - c) * ch + v / w * sh + c               # CoM position at end of step
        vT = w * (r - c) * sh + v * ch

        # base controller: capture point (DCM) deadbeat stepping, nominal model
        wn = self.w_nom
        xi_T = (r + v / wn) * np.exp(wn * self.T0)
        b_des = self.vd * self.T0 / (np.exp(wn * self.T0) - 1)
        L = np.clip(xi_T - b_des + dL + 0.015 * self._noise(), 0.0, 0.8)

        xs_new = self.xs + L
        h_new = self._height(xs_new)
        dh = h_new - self.hs
        v2 = vT ** 2 - 2 * G * LAM * dh + e              # energy lost climbing, push-off residual
        v_new = np.sqrt(np.maximum(v2, 0.0))
        r_new = rT - L

        pv = np.where((self._uniform() < self.push_prob) & (self.push_mag > 0),
                      np.sign(self._noise()) * (0.3 + 0.7 * self._uniform()) * self.push_mag, 0.0)
        if self.push_at is not None and self.t == self.push_at:
            pv = pv + self.push_val
        v_new = v_new + pv

        fall = (v_new < 0.05) | (np.abs(r_new) > 0.6) | (dh > 0.25) | (vT <= 0)
        alive = self.alive
        rew = 1.0 - 1.5 * np.abs(v_new - self.vd) \
              - 0.1 * (a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2) - 0.5 * a[:, 3] ** 2
        rew = np.where(alive, np.where(fall, -3.0, rew), 0.0)
        ok = alive & ~fall

        if self.record:
            self.log.append(dict(xs=self.xs[0], hs=self.hs[0], r=r[0], w=w[0], c=c[0], T=T[0],
                                 L=L[0], hn=h_new[0], zc=self.zc[0], v=v[0], ok=bool(ok[0]), pv=pv[0]))
        self.effort += np.where(ok, np.abs(e), 0)
        self.err += np.where(ok, np.abs(v_new - self.vd), 0)
        self.n_ok += ok
        self.xs = np.where(ok, xs_new, self.xs)
        self.hs = np.where(ok, h_new, self.hs)
        self.r = np.where(ok, r_new, self.r)
        self.v = np.where(ok, v_new, self.v)
        self.last_a = np.where(ok[:, None], a, self.last_a)
        self.alive = ok
        self.t += 1
        return self.obs(), rew


# ----------------------------------------------------------------------------
# Residual policy: small MLP, batched so many perturbed policies run at once
# ----------------------------------------------------------------------------
SIZES = (BipedEnv.OBS, 32, 32, BipedEnv.ACT)


def n_params():
    return sum(i * o + o for i, o in zip(SIZES[:-1], SIZES[1:]))


def init_theta(rng):
    th, parts = [], list(zip(SIZES[:-1], SIZES[1:]))
    for k, (i, o) in enumerate(parts):
        scale = 0.0 if k == len(parts) - 1 else 1 / np.sqrt(i)   # last layer = 0 -> start as pure base
        th += [rng.normal(0, scale, i * o) if scale else np.zeros(i * o), np.zeros(o)]
    return np.concatenate(th)


def forward(theta, obs):
    """theta [P,D], obs [P,E,I] -> actions [P,E,A] in [-1,1]."""
    x, k = obs, 0
    for li, (i, o) in enumerate(zip(SIZES[:-1], SIZES[1:])):
        W = theta[:, k:k + i * o].reshape(-1, i, o); k += i * o
        b = theta[:, k:k + o][:, None, :]; k += o
        x = np.einsum("pei,pio->peo", x, W) + b
        if li < len(SIZES) - 2:
            x = np.tanh(x)
    return np.tanh(x)


def rollout(env, theta):
    theta = np.atleast_2d(theta)
    obs = env.obs()
    total = np.zeros(env.N)
    for _ in range(env.max_steps):
        act = forward(theta, obs.reshape(env.P, env.E, -1)).reshape(env.N, -1)
        obs, rew = env.step(act)
        total += rew
        if not env.alive.any():
            break
    return total


# ----------------------------------------------------------------------------
# Training: Augmented Random Search + curriculum
# ----------------------------------------------------------------------------
def stage_params(rng, E, stage):
    pick = np.where(rng.random(E) < 0.6, stage, rng.integers(0, stage + 1, E))  # 60% current, rest replay
    g = lambda key: np.array([STAGES[s][key] for s in pick])
    return dict(slope_deg=g("slope"), stair_h=g("stair"), push_mag=g("push"))


def train(iters=300, seed=0, nd=24, E=32, nu=0.05, lr=0.02, top=12, promote=0.85, save="theta.npz"):
    rng = np.random.default_rng(seed)
    theta = init_theta(rng)
    env = BipedEnv(2 * nd, E, seed)
    ev = BipedEnv(1, 128, seed + 1)
    stage, stage_start, hist = 0, 0, []
    best_succ, best_theta, streak = -1.0, theta.copy(), 0
    t0 = time.time()
    for it in range(iters):
        env.reset(**stage_params(rng, E, stage))
        d = rng.normal(size=(nd, theta.size))
        R = rollout(env, np.concatenate([theta + nu * d, theta - nu * d])).reshape(2 * nd, E).mean(1)
        Rp, Rm = R[:nd], R[nd:]
        idx = np.argsort(-np.maximum(Rp, Rm))[:top]
        sigma = np.concatenate([Rp[idx], Rm[idx]]).std() + 1e-8
        theta = theta + lr / (top * sigma) * ((Rp[idx] - Rm[idx])[:, None] * d[idx]).sum(0)

        if it % 5 == 0:
            s = STAGES[stage]
            ev.reset(slope_deg=s["slope"], stair_h=s["stair"], push_mag=s["push"])
            rollout(ev, theta)
            succ = ev.alive.mean()
            hist.append((it, stage, succ))
            print(f"it {it:4d} | stage {stage} ({s['name']:6s}) | success {succ:.2f} | "
                  f"ret {R.mean():6.1f} | {time.time() - t0:5.1f}s")
            # keep a checkpoint; roll back if training collapses (ARS can be jumpy)
            if succ >= best_succ:
                best_succ, best_theta = succ, theta.copy()
            elif succ < best_succ - 0.25:
                theta = best_theta.copy()
                print("   collapse detected, rolled back to checkpoint")
            streak = streak + 1 if succ >= promote else 0
            if streak >= 2 and stage < len(STAGES) - 1:       # two good evals in a row -> promote
                stage, stage_start, best_succ, streak = stage + 1, it, -1.0, 0
    np.savez(save, theta=theta, hist=np.array(hist))
    return theta, np.array(hist)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
SCENARIOS = {
    "flat":       dict(slope_deg=0, stair_h=0, push_mag=0),
    "slope 10deg": dict(slope_deg=10, stair_h=0, push_mag=0),
    "stairs 10cm": dict(slope_deg=0, stair_h=0.10, push_mag=0),
    "rough":      dict(slope_deg=12, stair_h=0.15, push_mag=0),
    "rough+push": dict(slope_deg=12, stair_h=0.15, push_mag=0.5),
}


def evaluate(theta, n=500, seed=123):
    out = {}
    for name, kw in SCENARIOS.items():
        env = BipedEnv(1, n, seed)
        env.reset(randomize_amp=False, **kw)
        rollout(env, theta)
        out[name] = dict(success=env.alive.mean(), dist=env.xs.mean(),
                         speed_err=(env.err.sum() / max(env.n_ok.sum(), 1)),
                         effort_per_m=env.effort.sum() / max(env.xs.sum(), 1e-6))
    return out


def push_curve(theta, vals=np.linspace(-1.5, 1.5, 13), n=200, seed=7):
    res = []
    for v in vals:
        env = BipedEnv(1, n, seed)
        env.reset(push_at=10, push_val=v)
        rollout(env, theta)
        res.append(env.alive.mean())
    return vals, np.array(res)


def full_eval(theta, hist=None):
    base = np.zeros_like(theta)
    rb, rr = evaluate(base), evaluate(theta)
    print(f"\n{'scenario':14s} {'':3s}{'success':>16s}{'dist (m)':>16s}{'speed err':>16s}{'effort/m':>16s}")
    print(f"{'':14s}   {'base':>8s}{'resid':>8s}{'base':>8s}{'resid':>8s}{'base':>8s}{'resid':>8s}{'base':>8s}{'resid':>8s}")
    for k in rb:
        row = [rb[k]["success"], rr[k]["success"], rb[k]["dist"], rr[k]["dist"],
               rb[k]["speed_err"], rr[k]["speed_err"], rb[k]["effort_per_m"], rr[k]["effort_per_m"]]
        print(f"{k:14s}   " + "".join(f"{x:8.2f}" for x in row))
    vals, pb = push_curve(base)
    _, pr = push_curve(theta)

    fig, ax = plt.subplots(1, 3 if hist is not None else 2, figsize=(15, 4))
    i = 0
    if hist is not None:
        ax[0].plot(hist[:, 0], hist[:, 2], label="success (current stage)")
        ax[0].step(hist[:, 0], hist[:, 1] / (len(STAGES) - 1), "r--", where="post", label="stage / max")
        ax[0].set_title("Training curve"); ax[0].set_xlabel("iteration"); ax[0].legend()
        i = 1
    names = list(rb)
    x = np.arange(len(names))
    ax[i].bar(x - 0.2, [rb[k]["success"] for k in names], 0.4, label="base only")
    ax[i].bar(x + 0.2, [rr[k]["success"] for k in names], 0.4, label="base + residual")
    ax[i].set_xticks(x); ax[i].set_xticklabels(names, rotation=20); ax[i].set_title("Success rate"); ax[i].legend()
    ax[i + 1].plot(vals, pb, "o-", label="base only")
    ax[i + 1].plot(vals, pr, "o-", label="base + residual")
    ax[i + 1].set_title("Push recovery (flat)"); ax[i + 1].set_xlabel("velocity kick (m/s)"); ax[i + 1].legend()
    plt.tight_layout(); plt.savefig("results.png", dpi=120)
    print("saved results.png")


# ----------------------------------------------------------------------------
# Animation (GIF)
# ----------------------------------------------------------------------------
def _leg_ik(hip, ank, l1=0.5, l2=0.5):
    """Two-link leg, knee bends forward."""
    d = ank - hip
    n0 = max(np.hypot(*d), 1e-6)
    dist = np.clip(n0, 1e-3, l1 + l2 - 1e-3)
    u = d / n0
    a = (l1 ** 2 - l2 ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    return hip + a * u + h * np.array([-u[1], u[0]])


def _pick_episode(theta, scenario, n_try=300, steps=40):
    """Find a nice demo run: survives, gets pushed, crosses at least one stair."""
    env = None
    for sd in range(n_try):
        env = BipedEnv(1, 1, sd, max_steps=steps)
        env.reset(**SCENARIOS[scenario])
        env.record = True
        rollout(env, theta)
        pushes = sum(abs(s["pv"]) > 0.05 for s in env.log)
        stairs = np.abs(np.diff(env.H[0][:int(16 / DX)])).max() > 0.04
        if env.alive[0] and pushes >= 2 and stairs:
            break
    return env


def _build_frames(env, K=6):
    frames, swing0, tg = [], (-0.3, 0.0), 0.0
    pts = []
    for k, s in enumerate(env.log):
        pv_prev = env.log[k - 1]["pv"] if k > 0 else 0.0
        pts.append((s["xs"], s["hs"]))
        for j in range(K):
            f = j / K
            t = f * s["T"]
            w, c = s["w"], s["c"]
            r = (s["r"] - c) * np.cosh(w * t) + s["v"] / w * np.sinh(w * t) + c
            v = w * (s["r"] - c) * np.sinh(w * t) + s["v"] * np.cosh(w * t)
            sm = 3 * f ** 2 - 2 * f ** 3
            hip = (s["xs"] + r, s["hs"] + (s["hn"] - s["hs"]) * sm + s["zc"])
            swing = (swing0[0] + (s["xs"] + s["L"] - swing0[0]) * sm,
                     swing0[1] + (s["hn"] - swing0[1]) * sm + 0.10 * np.sin(np.pi * f))
            frames.append(dict(hip=hip, stance=(s["xs"], s["hs"]), swing=swing, par=k % 2, f=f, v=v,
                               xi=hip[0] + v / w, t=tg + t, k=k, npts=k + 1,
                               push=pv_prev if (j < 4 and abs(pv_prev) > 0.05) else 0.0,
                               push_new=(j == 0 and abs(pv_prev) > 0.05), fell=False))
        swing0 = (s["xs"], s["hs"])
        tg += s["T"]
        if not s["ok"]:
            for fr in frames[-(K - K // 2):]:
                fr["fell"] = True
            frames += [dict(frames[-1]) for _ in range(10)]
            break
    return frames, pts


def animate(theta, scenario="rough+push", out="walk.gif", K=6, fps=20):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap

    env = _pick_episode(theta, scenario)
    frames, pts = _build_frames(env, K)
    H, xs, vd = env.H[0], np.arange(M) * DX, env.vd[0]
    hips = np.array([fr["hip"] for fr in frames])

    FW, FH, VW = 9.0, 5.4, 7.5
    fig = plt.figure(figsize=(FW, FH), dpi=80, facecolor="#f4f6fa")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1], hspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.10)
    ax, av = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    pos = ax.get_position()
    VH = VW * (pos.height * FH) / (pos.width * FW)

    # ---- scene -----------------------------------------------------------
    ax.set_axis_off()
    ax.imshow(np.linspace(1, 0, 64)[:, None], extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto",
              cmap=LinearSegmentedColormap.from_list("sky", ["#f1f6fc", "#c9dcf1"]), zorder=0)
    ax.set_autoscale_on(False)
    ax.fill_between(xs, H - 10, H, color="#434c5e", lw=0, zorder=2)
    ax.plot(xs, H, color="#7fb069", lw=3.2, zorder=3, solid_capstyle="round")
    for xm in range(0, int(M * DX), 2):
        ax.text(xm, H[int(xm / DX)] - 0.22, f"{xm} m", ha="center", va="top", fontsize=7, color="#aab3c5", zorder=3)

    C_R, C_L, C_BODY = "#1d3557", "#5b8fd1", "#2b2d42"
    mk = lambda col, lw, z: ax.plot([], [], color=col, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)[0]
    legs = [mk(C_R, 9, 5), mk(C_L, 9, 4)]           # index 0 = parity 0 stance colour pair
    feet = [mk(C_R, 8, 5), mk(C_L, 8, 4)]
    arm_b, arm_f = mk("#9aa5b8", 6, 3), mk("#6b7a99", 6, 7)
    torso = mk(C_BODY, 19, 6)
    head = Circle((0, 0), 0.115, fc=C_BODY, ec="white", lw=1.5, zorder=7)
    ax.add_patch(head)
    visor = mk("#8ecae6", 4, 8)
    joints, = ax.plot([], [], "o", ms=5, mfc="white", mec=C_R, mew=1.3, zorder=9, ls="")
    com, = ax.plot([], [], "o", ms=11, mfc="#ff9f1c", mec="white", mew=1.6, zorder=10)
    cp, = ax.plot([], [], "D", ms=7, mfc="#2a9d8f", mec="white", mew=1.2, zorder=9)
    trail, = ax.plot([], [], color="#ff9f1c", lw=2.2, alpha=0.55, zorder=4)
    holds, = ax.plot([], [], "s", ms=5, mfc="#eceff4", mec="none", zorder=4)
    hud = ax.text(0.015, 0.95, "", transform=ax.transAxes, va="top", fontsize=9.5, color=C_BODY,
                  family="DejaVu Sans Mono", weight="bold")
    banner = ax.text(0.5, 0.86, "", transform=ax.transAxes, ha="center", fontsize=15, weight="bold", color="#d62828")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="#ff9f1c", mec="white", ms=9, label="CoM"),
        Line2D([], [], marker="D", ls="", mfc="#2a9d8f", mec="white", ms=7, label="capture point"),
        Line2D([], [], marker="s", ls="", mfc="#eceff4", mec="#434c5e", ms=6, label="footholds")],
        loc="upper right", ncol=3, fontsize=8, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")

    # ---- speed strip -----------------------------------------------------
    T_end = frames[-1]["t"] + 0.2
    av.set_facecolor("white")
    for sp in ("top", "right"):
        av.spines[sp].set_visible(False)
    av.set_xlim(0, T_end); av.set_ylim(0, 1.6)
    av.axhline(vd, ls="--", color="#8d99ae", lw=1)
    av.set_ylabel("speed (m/s)", fontsize=8); av.set_xlabel("time (s)", fontsize=8)
    av.tick_params(labelsize=8)
    vline, = av.plot([], [], color=C_R, lw=1.8)
    pmark, = av.plot([], [], "v", color="#d62828", ms=7)
    push_t = [fr["t"] for fr in frames if fr["push_new"]]
    push_v = [fr["v"] for fr in frames if fr["push_new"]]
    fig.text(0.07, 0.95, "Biped locomotion: capture-point base + residual RL", fontsize=13, weight="bold", color=C_BODY)
    fig.text(0.07, 0.915, f"reduced-order LIPM model | scenario: {scenario} | target speed {vd:.1f} m/s",
             fontsize=8.5, color="#6c757d")

    # ---- smoothed camera --------------------------------------------------
    cam = np.zeros((len(frames), 2))
    cx, cy = hips[0, 0] + 1.0, hips[0, 1]
    for i, h in enumerate(hips):
        cx += 0.18 * (h[0] + 1.0 - cx); cy += 0.10 * (h[1] - cy)
        cam[i] = (cx, cy)

    arrow = [None]
    ts = np.array([fr["t"] for fr in frames]); vs = np.array([fr["v"] for fr in frames])

    def update(i):
        fr = frames[i]
        hip, st, sw = np.array(fr["hip"]), np.array(fr["stance"]), np.array(fr["swing"])
        for leg, foot, p, idx in ((legs, feet, st, fr["par"]), (legs, feet, sw, 1 - fr["par"])):
            ank = p + np.array([-0.02, 0.09])
            knee = _leg_ik(hip, ank)
            leg[idx].set_data([hip[0], knee[0], ank[0]], [hip[1], knee[1], ank[1]])
            foot[idx].set_data([ank[0], p[0] + 0.15], [ank[1], p[1] + 0.02])
        ks = [_leg_ik(hip, st + np.array([-0.02, 0.09])), _leg_ik(hip, sw + np.array([-0.02, 0.09]))]
        joints.set_data([k_[0] for k_ in ks] + [st[0] - 0.02, sw[0] - 0.02],
                        [k_[1] for k_ in ks] + [st[1] + 0.09, sw[1] + 0.09])
        lean = np.clip(0.10 + 0.12 * (fr["v"] - 0.8), 0.0, 0.3)
        sh = hip + 0.55 * np.array([np.sin(lean), np.cos(lean)])
        torso.set_data([hip[0], sh[0]], [hip[1], sh[1]])
        hc = sh + 0.17 * np.array([np.sin(lean), np.cos(lean)])
        head.center = tuple(hc)
        visor.set_data([hc[0] + 0.02, hc[0] + 0.09], [hc[1] + 0.02, hc[1] + 0.02])
        sgn = 1 if fr["par"] == 0 else -1
        th = sgn * 0.55 * (1 - 2 * fr["f"])
        for arm, a_ in ((arm_f, th), (arm_b, -th)):
            el = sh + 0.28 * np.array([np.sin(a_), -np.cos(a_)])
            hand = el + 0.26 * np.array([np.sin(a_ + 0.5), -np.cos(a_ + 0.5)])
            arm.set_data([sh[0], el[0], hand[0]], [sh[1] - 0.04, el[1], hand[1]])
        com.set_data([hip[0]], [hip[1]])
        cp.set_data([fr["xi"]], [H[int(np.clip(fr["xi"] / DX, 0, M - 1))] + 0.04])
        a0 = max(0, i - 70)
        trail.set_data(hips[a0:i + 1, 0], hips[a0:i + 1, 1])
        P = np.array(pts[:fr["npts"]])
        holds.set_data(P[:, 0], P[:, 1] + 0.01)

        hud.set_text(f"step {fr['k'] + 1:02d}   t = {fr['t']:4.1f} s\nspeed {fr['v']:.2f} m/s")
        banner.set_text("FELL" if fr["fell"] else (f"PUSH {fr['push']:+.2f} m/s" if fr["push"] else ""))
        if arrow[0] is not None:
            arrow[0].remove(); arrow[0] = None
        if fr["push"]:
            d = np.sign(fr["push"])
            y = hip[1] + 0.35
            arrow[0] = ax.annotate("", xy=(hip[0] - d * 0.45, y), xytext=(hip[0] - d * 1.25, y),
                                   arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=3.5, mutation_scale=20), zorder=11)
        cx_, cy_ = cam[i]
        ax.set_xlim(cx_ - VW / 2, cx_ + VW / 2)
        ymin = cy_ - 1.15
        ax.set_ylim(ymin, ymin + VH)
        vline.set_data(ts[:i + 1], vs[:i + 1])
        done = [(t_, v_) for t_, v_ in zip(push_t, push_v) if t_ <= fr["t"]]
        pmark.set_data([d[0] for d in done], [d[1] for d in done])

    FuncAnimation(fig, update, frames=len(frames), blit=False).save(out, writer=PillowWriter(fps=fps))
    print(f"saved {out} ({len(frames)} frames)")
    return frames


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval", "animate", "all"])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    hist = None
    if a.cmd in ("train", "all"):
        theta, hist = train(a.iters, a.seed)
    else:
        z = np.load("theta.npz"); theta, hist = z["theta"], z["hist"]
    if a.cmd in ("eval", "all"):
        full_eval(theta, hist)
    if a.cmd in ("animate", "all"):
        animate(theta)