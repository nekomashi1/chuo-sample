"""Synthesise the 15 s showreel soundtrack (120 BPM, A minor), locked to the picture cuts.
python3 audio.py  ->  showreel-audio.wav (48 kHz, 16-bit stereo)
Requires numpy + scipy. Deterministic (fixed seed)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import wave

SR = 48000
DUR = 15.0
N = int(SR * DUR)
rng = np.random.default_rng(2026)
BEAT = 0.5

drums = np.zeros((N, 2))
music = np.zeros((N, 2))   # ducked by the kick
fx = np.zeros((N, 2))
send = np.zeros((N, 2))    # reverb send


def tt(d):
    return np.arange(int(d * SR)) / SR


def place(bus, sig, t0, gain=1.0, pan=0.0, rev=0.0):
    """Add mono/stereo sig at time t0 with constant-power pan (-1..1)."""
    i = int(round(t0 * SR))
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)], 1) * np.sqrt(2)
    j = min(N, i + len(sig))
    if j <= i:
        return
    bus[i:j] += sig[: j - i] * gain
    if rev:
        send[i:j] += sig[: j - i] * gain * rev


def filt(x, kind, f, order=2):
    if np.isscalar(f):
        f = min(f, SR / 2 * 0.95)
    else:
        f = [min(v, SR / 2 * 0.95) for v in f]
    return sosfilt(butter(order, f, kind, fs=SR, output='sos'), x, axis=0)


def svf_sweep(x, f0, f1, q=0.7, mode='lp', curve=2.0):
    """State-variable filter with exponential cutoff sweep f0 -> f1."""
    n = len(x)
    s = np.linspace(0, 1, n) ** curve
    fc = f0 * (f1 / f0) ** s
    g = np.tan(np.pi * np.minimum(fc, SR * 0.45) / SR)
    k = 1 / q
    ic1 = ic2 = 0.0
    out = np.empty(n)
    for i in range(n):
        gi = g[i]
        v1 = (ic1 + gi * (x[i] - ic2)) / (1 + gi * (gi + k))
        v2 = ic2 + gi * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        out[i] = v2 if mode == 'lp' else (v1 if mode == 'bp' else x[i] - k * v1 - v2)
    return out


def noise(d):
    return rng.standard_normal(int(d * SR))


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(f, d, harm=24, detune=0.0):
    t = tt(d)
    out = np.zeros_like(t)
    for h in range(1, harm + 1):
        if f * h > SR / 2 * 0.9:
            break
        out += np.sin(2 * np.pi * f * (1 + detune) * h * t) / h
    return out * 0.6


# ---------------- instruments ----------------
def kick(d=0.45, punch=1.0):
    t = tt(d)
    f = 44 + 120 * np.exp(-t * 32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.5)
    click = filt(noise(d), 'highpass', 2500) * np.exp(-t * 400) * 0.35
    return np.tanh((body + click) * 1.6 * punch)


def clap(d=0.25):
    t = tt(d)
    n = filt(noise(d), 'bandpass', [900, 3200])
    env = np.zeros_like(t)
    for k, o in enumerate([0, 0.011, 0.022]):
        env += (t >= o) * np.exp(-np.maximum(t - o, 0) * (160 if k < 2 else 22))
    return n * env * 0.8


def snare(d=0.16, tone=190):
    t = tt(d)
    n = filt(noise(d), 'bandpass', [1200, 8000]) * np.exp(-t * 26)
    b = np.sin(2 * np.pi * tone * t) * np.exp(-t * 30)
    return (n * 0.8 + b * 0.5)


def hat(d=0.05, open_=False):
    t = tt(0.3 if open_ else d)
    return filt(noise(len(t) / SR), 'highpass', 7500) * np.exp(-t * (14 if open_ else 90)) * 0.5


def crash(d=1.6):
    t = tt(d)
    n = filt(noise(d), 'highpass', 3500)
    n += filt(noise(d), 'bandpass', [5000, 9000]) * 0.6
    return n * np.exp(-t * 2.6) * 0.5


def sub_boom(d=1.0, f0=62, f1=32):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t * 5)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(np.sin(ph) * np.exp(-t * 3.2) * 1.4)


def pluck(f, d=0.25, bright=1.0):
    t = tt(d)
    s = np.sin(2 * np.pi * f * t) + 0.35 * bright * np.sin(4 * np.pi * f * t) + 0.12 * bright * np.sin(6 * np.pi * f * t)
    return s * np.exp(-t * 16) * np.minimum(1, t * 800)


def blip(f, d=0.08):
    t = tt(d)
    fr = f * (1 + 0.5 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 45) * np.minimum(1, t * 2000)


def whoosh(d, f0, f1, q=1.4, curve=1.0, shape='swell'):
    t = tt(d)
    x = svf_sweep(noise(d), f0, f1, q=q, mode='bp', curve=curve)
    if shape == 'swell':
        env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5
    elif shape == 'rise':
        env = (t / d) ** 2.2
    else:
        env = np.exp(-t * 6)
    return x * env


def riser(d, f0, f1):
    """Noise + pitched saw rising, ends with full level (cut hard at the drop)."""
    t = tt(d)
    x = svf_sweep(noise(d), 300, 9000, q=0.9, mode='bp', curve=1.6) * (t / d) ** 2
    fr = f0 * (f1 / f0) ** ((t / d) ** 1.5)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * (t / d) ** 1.8 * 0.35
    return x + tone


# ---------------- arrangement ----------------
# chords per 2 s bar (A minor): Am  F  C  G  Am  F/G   -> final A
CHORDS = [(2, [57, 60, 64]), (4, [53, 57, 60]), (6, [55, 60, 64]), (8, [55, 59, 62]), (10, [57, 60, 64]),
          (12, [53, 57, 60]), (13, [55, 59, 62])]
BASS = [(2, 33), (4, 29), (6, 36), (8, 31), (10, 33), (12, 29), (13, 31)]
DROP = 13.875  # everything stops here, hit lands at 14.0

# --- intro (0 - 2 s)
place(fx, blip(1320, 0.18) * 0.5, 0.04, pan=0, rev=0.5)
place(fx, sub_boom(0.6, 90, 45) * 0.35, 0.04)
zt = tt(0.44)
zf = 180 * (2400 / 180) ** ((zt / 0.44) ** 2.2)
zip_ = np.sin(2 * np.pi * np.cumsum(zf) / SR) * (zt / 0.44) ** 1.5 * 0.18
place(fx, zip_ + whoosh(0.44, 400, 6000, curve=2.0, shape='rise') * 0.25, 0.34, rev=0.3)
place(drums, kick(0.35, 0.8) * 0.7, 0.78)
place(fx, hat(0.03) * 1.2, 0.78, rev=0.4)
for j in range(12):
    place(fx, blip(2400 + j * 90, 0.03) * 0.12, 0.8 + j * 0.022, pan=(-0.7 if j % 2 else 0.7))
place(fx, whoosh(0.6, 200, 2500, q=1.0) * 0.35, 0.9, rev=0.3)
place(fx, riser(1.0, 110, 440) * 0.55, 1.0)
rc = crash(1.0)[::-1] * 0.6  # reverse cymbal into the drop
place(fx, rc, 1.0)

# --- groove (2.0 - 13.875)
b = 2.0
while b < DROP - 1e-6:
    beat_in_bar = int(round((b - 2) / BEAT)) % 4
    place(drums, kick(), b, 0.95)
    if beat_in_bar in (1, 3) and b < 12.0:
        place(drums, clap(), b, 0.55, rev=0.25)
    if b + 0.25 < DROP:
        place(drums, hat(open_=(b >= 8 and beat_in_bar == 3)), b + 0.25, 0.35, pan=0.25)
    if b >= 6.0 and b < 12.0:
        for s in (0.125, 0.375):
            place(drums, hat(0.03), b + s, 0.16, pan=-0.35)
    b += BEAT

# pad + bass, ducked later
for k, (t0, notes) in enumerate(CHORDS):
    t1 = CHORDS[k + 1][0] if k + 1 < len(CHORDS) else DROP
    d = t1 - t0
    pad = np.zeros(int(d * SR))
    for m in notes:
        for dt in (-0.004, 0.004):
            pad += saw(midi(m), d, 16, dt)
    pad = filt(pad, 'lowpass', 1800 if t0 < 12 else 3200)
    pt = tt(d)
    pad *= np.minimum(1, pt / 0.05) * np.minimum(1, (d - pt) / 0.03)
    left = pad * 0.9 + np.roll(pad, 240) * 0.1
    right = np.roll(pad, 480)
    place(music, np.stack([left, right], 1) * 0.045, t0, rev=0.6)
for k, (t0, m) in enumerate(BASS):
    t1 = BASS[k + 1][0] if k + 1 < len(BASS) else DROP
    s = t0
    while s < t1 - 1e-6:
        dd = 0.24
        n = saw(midi(m), dd, 30) + 0.6 * np.sin(2 * np.pi * midi(m) * tt(dd))
        n = svf_sweep(n, 2400, 180, q=1.1, curve=0.5)
        env = np.exp(-tt(dd) * 5) * np.minimum(1, tt(dd) * 400) * np.minimum(1, (dd - tt(dd)) * 300)
        place(music, np.tanh(n * env * 1.4) * 0.34, s)
        s += 0.25

# --- scene 2 accents: a stab per word, the WEIGHT impact
for i, t0 in enumerate([2.0, 2.5, 3.0, 3.5]):
    notes = [69, 72, 76] if i != 2 else [57, 64, 69]
    st = sum(pluck(midi(m), 0.3) for m in notes) * 0.12
    place(fx, st, t0, pan=[-0.3, 0.3, 0, 0][i], rev=0.4)
for i in range(6):   # TIMING letters landing
    place(fx, blip(900 + i * 110, 0.05) * 0.14, 2.0 + i * 0.03 + 0.05, pan=-0.6 + i * 0.24)
for j in range(13):  # SPACING keys
    place(fx, blip(1800, 0.025) * 0.08, 2.5 + 0.04 + j * 0.018, pan=-0.8 + j * 0.13)
place(fx, whoosh(0.4, 300, 3000, q=2) * 0.3, 2.56, pan=0.4)
place(fx, whoosh(0.1, 2000, 200, q=1.2, shape='rise') * 0.5, 3.0)
place(drums, sub_boom(1.1, 70, 30) * 0.9, 3.1)
place(drums, kick(0.5, 1.4) * 0.8, 3.1)
place(fx, filt(noise(0.35), 'lowpass', 900) * np.exp(-tt(0.35) * 12) * 0.6, 3.1, rev=0.5)
place(fx, whoosh(0.35, 400, 5000, q=1.2, curve=1.2) * 0.45, 3.82, rev=0.3)   # strips

# --- scene 3: morph swishes, collapse suck
for t0 in (4.5, 5.0, 5.5):
    place(fx, whoosh(0.3, 700, 5200, q=2.5, curve=0.7) * 0.32, t0 - 0.02, pan=0.2, rev=0.3)
    place(fx, pluck(midi(81), 0.4, 0.4) * 0.08, t0, rev=0.6)
st = tt(0.24)
suck = (st / 0.24) ** 3
place(fx, (svf_sweep(noise(0.24), 200, 7000, q=1.5, mode='bp', curve=2) * suck * 0.6
           + np.sin(2 * np.pi * np.cumsum(120 * (8 ** (st / 0.24) ** 2)) / SR) * suck * 0.2), 5.76)

# --- scene 4: burst, arp shimmer, wind
place(drums, sub_boom(1.0, 80, 34) * 0.7, 6.0)
place(fx, crash(1.4) * 0.7, 6.0, rev=0.5)
place(fx, filt(noise(0.5), 'lowpass', 2500) * np.exp(-tt(0.5) * 9) * 0.5, 6.0, rev=0.6)
arp = [76, 79, 84, 88, 91, 88, 84, 79]
i = 0
s = 6.25
while s < 7.75:
    place(fx, pluck(midi(arp[i % len(arp)]), 0.22, 0.6) * 0.07, s, pan=np.sin(i * 0.9) * 0.7, rev=0.7)
    s += 0.125
    i += 1
place(fx, whoosh(0.3, 300, 900, q=0.6) * 0.3, 6.8)
place(fx, blip(1760, 0.3) * 0.1, 7.4, rev=0.8)
place(fx, whoosh(0.5, 250, 6000, q=0.8, curve=1.3) * 0.6, 7.66, pan=0.5, rev=0.3)

# --- scene 5: swell, morph accent, dolly riser
place(fx, whoosh(0.8, 3000, 400, q=0.8) * 0.25, 7.9, rev=0.5)
place(drums, kick(0.5, 1.2) * 0.4, 9.0)
place(fx, crash(0.9) * 0.3, 9.0, rev=0.4)
place(fx, whoosh(0.45, 500, 4000, q=2) * 0.3, 8.95, pan=-0.4)
place(fx, riser(0.55, 220, 880) * 0.5, 9.5)

# --- scene 6: UI sounds
for j in range(8):
    place(fx, blip(660 * 2 ** (j * 2 / 12), 0.07) * 0.14, 10.15 + j * 0.05, pan=-0.6 + j * 0.17, rev=0.25)
dt_ = tt(0.8)
place(fx, np.sin(2 * np.pi * np.cumsum(500 + 700 * (1 - np.exp(-dt_ * 5))) / SR) * np.exp(-dt_ * 4) * 0.05, 10.3, pan=0.5, rev=0.4)
for j in range(10):
    place(fx, blip(3000, 0.015) * 0.05, 10.35 + j * 0.06 * (1 + j * 0.15), pan=0.6)
place(fx, whoosh(0.5, 400, 1400, q=1.0) * 0.18, 10.85, pan=0.7)
click = filt(noise(0.02), 'bandpass', [2000, 6000]) * np.exp(-tt(0.02) * 300)
place(fx, click * 0.9, 11.4, pan=0.5, rev=0.3)
place(fx, blip(1400, 0.05) * 0.2, 11.4, pan=0.5, rev=0.3)
place(fx, riser(0.45, 300, 1200) * 0.55, 11.55)

# --- scene 7: accelerating build
for k in range(8):
    place(drums, snare(), 12.0 + k * 0.125, 0.25 + k * 0.02, rev=0.2)
for k in range(14):
    place(drums, snare(0.1, 200 + k * 12), 13.0 + k * 0.0625, 0.3 + k * 0.03, rev=0.2)
place(fx, riser(1.875, 110, 880) * 0.55, 12.0)
for c in [12.0, 12.25, 12.5, 12.75, 13.0, 13.125, 13.25, 13.375, 13.5, 13.625, 13.75, 13.8125]:
    g = filt(noise(0.03), 'bandpass', [1500, 9000]) * np.exp(-tt(0.03) * 120)
    g = np.round(g * 6) / 6  # crushed
    place(fx, g * 0.25, c, pan=rng.uniform(-0.8, 0.8))

# --- the drop gap: hard silence 13.875 - 14.0
for bus in (drums, music, fx, send):
    a, z = int(DROP * SR), int(14.0 * SR)
    fade = int(0.004 * SR)
    bus[a - fade:a] *= np.linspace(1, 0, fade)[:, None]
    bus[a:z] = 0

# --- 14.0 final hit
place(drums, kick(0.6, 1.5), 14.0)
place(drums, sub_boom(1.0, 70, 30) * 1.0, 14.0)
place(fx, crash(1.0) * 0.9, 14.0, rev=0.5)
fin = np.zeros(int(1.0 * SR))
for m in [45, 52, 57, 60, 64, 69]:
    for dt in (-0.006, 0.0, 0.006):
        fin += saw(midi(m), 1.0, 20, dt)
fin = filt(fin, 'lowpass', 4500) * np.exp(-tt(1.0) * 2.2) * np.minimum(1, tt(1.0) * 400)
place(music, np.stack([fin, np.roll(fin, 300)], 1) * 0.06, 14.0, rev=0.7)
place(fx, pluck(midi(81), 0.9, 0.3) * 0.18, 14.02, rev=0.9)
place(fx, pluck(midi(88), 0.9, 0.3) * 0.1, 14.3, pan=0.4, rev=0.9)

# ---------------- mix ----------------
# sidechain: duck music on every kick
kicks = [2.0 + i * BEAT for i in range(int((DROP - 2.0) / BEAT) + 1)] + [3.1, 14.0]
t = np.arange(N) / SR
duck = np.ones(N)
for k in kicks:
    m = t >= k
    duck[m] = np.minimum(duck[m], 1 - 0.7 * np.exp(-(t[m] - k) * 11))
music *= duck[:, None]

# reverb: exponentially decaying stereo noise IR
ir_t = tt(1.4)
ir = np.stack([rng.standard_normal(len(ir_t)), rng.standard_normal(len(ir_t))], 1) * np.exp(-ir_t * 4.2)[:, None]
ir = filt(ir, 'lowpass', 6000)
ir /= np.sqrt((ir ** 2).sum(0))
rev = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1)
rev = filt(rev, 'highpass', 200) * 0.9
for a, z in [(DROP, 14.0)]:
    rev[int(a * SR):int(z * SR)] *= 0.05

mix = drums * 0.5 + music * 0.85 + fx * 0.8 + rev * 0.8
mix = filt(mix, 'highpass', 24)
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.6) / np.tanh(1.6)
# tail fade
fo = int(0.25 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
mix /= np.abs(mix).max() / 0.72

pcm = (mix * 32767).astype('<i2')
with wave.open('showreel-audio.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote showreel-audio.wav', pcm.shape, 'rms', float(np.sqrt((mix ** 2).mean())))
