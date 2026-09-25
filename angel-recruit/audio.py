"""エンゼルガーデン幼稚園 先生募集リールの音（30秒・120BPM・ハ長調）
python3 audio.py -> reel-audio.wav (48kHz / 16bit / stereo)
ウクレレ（カープラス・ストロング）、鉄琴のメロディ、ピチカートのベース、軽いリズム、
画面の出来事に合わせた効果音。numpy + scipy。乱数は固定。"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import wave

SR = 48000
DUR = 30.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
BEAT = 0.5
BAR = 2.0

music = np.zeros((N, 2))
drums = np.zeros((N, 2))
fx = np.zeros((N, 2))
send = np.zeros((N, 2))


def tt(d):
    return np.arange(int(d * SR)) / SR


def place(bus, sig, t0, gain=1.0, pan=0.0, rev=0.0):
    i = int(round(t0 * SR))
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)], 1) * np.sqrt(2)
    j = min(N, i + len(sig))
    if j <= i or i < 0:
        return
    bus[i:j] += sig[: j - i] * gain
    if rev:
        send[i:j] += sig[: j - i] * gain * rev


def filt(x, kind, f, order=2):
    return sosfilt(butter(order, f, kind, fs=SR, output='sos'), x, axis=0)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def noise(d):
    return rng.standard_normal(int(d * SR))


# ---------------- instruments ----------------
def ks(f, d, damp=0.994, bright=0.6):
    """Karplus-Strong plucked string, vectorised one period at a time."""
    n = int(d * SR)
    P = max(2, int(round(SR / f)))
    buf = rng.uniform(-1, 1, P)
    buf = bright * buf + (1 - bright) * np.convolve(buf, np.ones(4) / 4, 'same')
    out = np.empty(n)
    k = 0
    while k < n:
        m = min(P, n - k)
        out[k:k + m] = buf[:m]
        buf = damp * 0.5 * (buf + np.roll(buf, 1))
        k += P
    env = np.minimum(1, tt(d) * 3000)
    return out * env


UKE = {'C': [67, 60, 64, 72], 'G': [67, 62, 67, 71], 'Am': [69, 60, 64, 69], 'F': [69, 60, 65, 69]}


def strum(chord, d=0.9, up=False, vel=1.0):
    notes = UKE[chord][::-1] if up else UKE[chord]
    out = np.zeros(int((d + 0.06) * SR))
    for k, m in enumerate(notes):
        s = ks(midi(m), d, damp=0.992, bright=0.5 if up else 0.7) * (0.7 if up else 1.0)
        o = int(k * 0.011 * SR)
        out[o:o + len(s)] += s
    return filt(out, 'highpass', 150) * vel * 0.22


def glock(f, d=1.2, vel=1.0):
    t = tt(d)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 3.2)
         + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 9)
         + 0.15 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 18))
    return s * np.minimum(1, t * 3000) * vel * 0.2


def pizz(f, d=0.35):
    t = tt(d)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    return s * np.exp(-t * 9) * np.minimum(1, t * 900) * 0.45


def kick(d=0.25):
    t = tt(d)
    f = 55 + 90 * np.exp(-t * 40)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14) * 0.8


def snap(d=0.12):
    t = tt(d)
    return filt(noise(d), 'bandpass', [1500, 5000]) * np.exp(-t * 45) * 0.45


def shaker(d=0.06):
    t = tt(d)
    return filt(noise(d), 'highpass', 6000) * np.sin(np.pi * np.clip(t / d, 0, 1)) * 0.18


def boing(f0=260, d=0.3):
    t = tt(d)
    fr = f0 * (1 + 1.2 * (t / d)) + 25 * np.sin(2 * np.pi * 22 * t)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 7) * np.minimum(1, t * 800) * 0.35


def pop(f=900, d=0.08):
    t = tt(d)
    fr = f * (1 + 1.5 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 40) * np.minimum(1, t * 3000) * 0.3


def swish(d=0.45, f0=500, f1=4000):
    t = tt(d)
    x = noise(d)
    out = np.zeros_like(x)
    segs = 12
    for k in range(segs):   # stepped band sweep (cheap)
        a, b = int(k * len(x) / segs), int((k + 1) * len(x) / segs)
        fc = f0 * (f1 / f0) ** (k / (segs - 1))
        out[a:b] = filt(x, 'bandpass', [fc * 0.7, min(fc * 1.4, 20000)])[a:b]
    return out * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5 * 0.25


def chirp(d=0.09, f0=3200, f1=4600):
    t = tt(d)
    fr = f0 + (f1 - f0) * np.sin(np.pi * t / d)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.sin(np.pi * t / d) * 0.12


def scribble(d):
    t = tt(d)
    x = filt(noise(d), 'bandpass', [1800, 5500])
    am = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 6.5 * t + np.sin(2 * np.pi * 1.3 * t)))
    return x * am * np.minimum(1, t * 30) * np.minimum(1, (d - t) * 30) * 0.12


# ---------------- arrangement ----------------
PROG = ['C', 'G', 'Am', 'F']
ROOT = {'C': 48, 'G': 43, 'Am': 45, 'F': 41}
# chord per bar; last bars resolve: ... F G | C
chords = [PROG[b % 4] for b in range(15)]
chords[12], chords[13], chords[14] = 'F', 'G', 'C'
END = 28.0   # final chord lands here

# ukulele island strum: D . D U . U D U
STRUM = [(0, False, 1.0), (1, False, .8), (1.5, True, .6), (2.5, True, .6), (3, False, .8), (3.5, True, .6)]
for b, ch in enumerate(chords):
    t0 = b * BAR
    if t0 >= END:
        break
    for beat, up, v in STRUM:
        t = t0 + beat * BEAT
        if t >= END:
            continue
        v2 = v * (0.6 if t < 1.0 else 1.0)
        place(music, strum(ch, 0.9, up, v2), t, pan=-0.25, rev=0.25)
place(music, strum('C', 2.2, False, 1.3), END, pan=-0.2, rev=0.5)
place(music, strum('C', 2.2, True, .8), END + 0.06, pan=0.2, rev=0.5)

# bass from 3.5 s
for b, ch in enumerate(chords):
    t0 = b * BAR
    r = ROOT[ch]
    for beat, iv in [(0, 0), (1.5, 7), (2, 0), (3, 7), (3.5, 12)]:
        t = t0 + beat * BEAT
        if 3.5 <= t < END:
            place(music, pizz(midi(r + iv)), t, 0.9)
place(music, pizz(midi(36), 1.2) * 1.2, END)

# light drums from 3.5 s, dropped for the message (23.5-25)
for i in range(int(END / BEAT)):
    t = i * BEAT
    if t < 3.5 or 23.5 <= t < 25.0:
        continue
    beat = i % 4
    if beat in (0, 2):
        place(drums, kick(), t, 0.8)
    if beat in (1, 3):
        place(drums, snap(), t, 0.7, pan=0.1, rev=0.2)
    place(drums, shaker(), t + 0.25, 1.0, pan=0.4)
    place(drums, shaker(), t, 0.6, pan=0.4)

# glockenspiel melody from 3.5 s (8th notes, None = rest)
MEL = {
    'C': [76, None, 79, None, 81, 79, 76, None],
    'G': [74, None, 79, None, 83, 81, 79, None],
    'Am': [72, None, 76, None, 81, 79, 76, None],
    'F': [72, None, 77, None, 81, 79, 77, 74],
}
for b, ch in enumerate(chords):
    t0 = b * BAR
    if t0 < 2.0 or t0 >= END:
        continue
    for k, m in enumerate(MEL[ch]):
        t = t0 + k * 0.25
        if m is None or t < 3.5 or t >= END:
            continue
        if 23.5 <= t < 25.0 and k % 2:
            continue
        place(music, glock(midi(m + 12), 1.0, 0.8), t, pan=0.3, rev=0.4)
place(music, glock(midi(84), 2.0, 1.2), END, pan=0.2, rev=0.7)
place(music, glock(midi(91), 2.0, 0.8), END + 0.25, pan=-0.2, rev=0.7)
place(music, glock(midi(96), 2.0, 0.7), END + 0.5, pan=0.1, rev=0.7)

# ---------------- sound effects (match reel.html timings) ----------------
# S1: ball bounces, sun, text, kids peeking
for t, f in [(0.35, 240), (0.75, 300), (1.0, 360)]:
    place(fx, boing(f, 0.3), t, pan=0)
for k, m in enumerate([72, 76, 79, 84]):
    place(fx, glock(midi(m + 12), 1.0, 0.8), 1.35 + k * 0.08, pan=-0.3 + k * 0.2, rev=0.6)
for t in [0.55, 0.9, 1.2, 1.5]:
    place(fx, pop(1000), t, 0.8)
for k, t in enumerate([2.1, 2.3, 2.5]):
    place(fx, pop(600 + k * 200, 0.12), t + 0.12, 0.9, pan=-0.5 + k * 0.5)
# scene wipes
for b in [3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.0]:
    place(fx, swish(0.6, 600, 5000), b - 0.32, 0.9, rev=0.3)
# S2
place(fx, chirp(), 3.9, pan=-0.6); place(fx, chirp(0.08, 3600, 5000), 4.02, pan=-0.6)
place(fx, chirp(), 5.1, pan=-0.2); place(fx, chirp(0.08, 3600, 5000), 5.22, pan=-0.2)
place(fx, pop(800, 0.12), 4.5, 1.0, pan=-0.4)
place(fx, pop(1100, 0.12), 5.9, 1.0, pan=0.4)
place(fx, glock(midi(88), 0.8, 0.5), 5.95, pan=0.4, rev=0.5)
# S3: ball bounces, jump take-offs
k = 1
while True:
    t = 7.5 + k * np.pi / 6
    if t > 11.3:
        break
    place(fx, boing(420, 0.18) * 0.5, t, pan=0.5 - (t - 7.5) * 0.3)
    k += 1
k = 0
while True:
    t = 7.5 + k * 2 * np.pi / 5
    if t > 11.3:
        break
    place(fx, boing(300, 0.25) * 0.6, t, pan=0.2)
    k += 1
for k in range(8):
    place(fx, pop(900 + k * 60, 0.06) * 0.5, 7.7 + k * 0.07, pan=-0.8 + k * 0.22)
# S4: crayon strokes
for a, b in [(0.3, 0.75), (0.75, 1.06), (1.05, 1.5), (2.2, 2.6)]:
    place(fx, scribble(b - a), 11.5 + a, pan=0.1)
for k in range(6):
    place(fx, pop(1200 + k * 80, 0.06) * 0.6, 11.5 + 1.55 + k * 0.07, pan=0.1)
for k, t in enumerate([2.7, 2.85, 3.0]):
    place(fx, glock(midi(91 + k * 2), 0.6, 0.6), 11.5 + t, pan=-0.4 + k * 0.4, rev=0.5)
# S5: lid, food, いただきます
place(fx, pop(500, 0.15), 15.8, 1.2)
place(fx, swish(0.4, 1500, 6000), 15.85, 0.6, pan=0.5)
for k, t in enumerate([0.55, 0.7, 0.85, 1.0]):
    place(fx, pop(800 + k * 150, 0.08), 15.5 + t, 0.9, pan=-0.3 + k * 0.2)
for k, m in enumerate([72, 76, 79]):
    place(fx, glock(midi(m + 12), 1.0, 0.7), 17.0 + k * 0.1, rev=0.6)
# S6: bubble, twinkles
place(fx, pop(900, 0.12), 20.3, 1.0, pan=-0.4)
r = np.random.default_rng(61)
for k in range(14):
    t = 19.5 + 1 + r.uniform() * 2
    place(fx, glock(midi(int(r.choice([88, 91, 93, 96]))), 0.8, 0.35), t, pan=r.uniform(-.8, .8), rev=0.7)
# S7: message pops, できた！ fanfare + confetti
for t in [23.7, 24.6, 24.9]:
    place(fx, pop(900, 0.08) * 0.7, t)
place(fx, filt(noise(0.25), 'highpass', 1500) * np.exp(-tt(0.25) * 18) * 0.5, 24.05, rev=0.3)
for k, m in enumerate([72, 76, 79, 84, 88]):
    place(fx, glock(midi(m + 12), 1.2, 0.9), 24.05 + k * 0.06, pan=-0.4 + k * 0.2, rev=0.6)
for k in range(5):
    place(fx, pop(1300 + k * 100, 0.06) * 0.6, 23.5 + 0.6 + k * 0.08, pan=-0.5 + k * 0.25)
# S8: emblem, ribbon, CTA, cast
place(fx, glock(midi(96), 1.5, 0.8), 27.05, rev=0.7)
place(fx, swish(0.4, 800, 4000), 27.6, 0.7)
place(fx, pop(1000, 0.1), 27.8, 0.8)
place(fx, pop(1200, 0.1), 28.2, 0.8)
for k in range(5):
    place(fx, pop(700 + k * 120, 0.1) * 0.8, 27.9 + k * 0.08, pan=-0.7 + k * 0.35)

# ---------------- mix ----------------
t = np.arange(N) / SR
ir_t = tt(1.6)
ir = np.stack([rng.standard_normal(len(ir_t)), rng.standard_normal(len(ir_t))], 1) * np.exp(-ir_t * 3.5)[:, None]
ir = filt(ir, 'lowpass', 7000)
ir /= np.sqrt((ir ** 2).sum(0))
rev = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1)
rev = filt(rev, 'highpass', 250)

mix = music * 1.0 + drums * 0.8 + fx * 0.9 + rev * 0.7
mix = filt(mix, 'highpass', 35)
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.3) / np.tanh(1.3)
fi = int(0.02 * SR)
mix[:fi] *= np.linspace(0, 1, fi)[:, None]
fo = int(1.2 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
mix /= np.abs(mix).max() / 0.72

pcm = (mix * 32767).astype('<i2')
with wave.open('reel-audio.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote reel-audio.wav', pcm.shape)
