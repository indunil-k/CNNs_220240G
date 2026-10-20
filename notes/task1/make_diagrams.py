"""Generates the Task 1 study diagrams (SVG) in this folder. Run: python notes/task1/make_diagrams.py"""
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def fmt(n):
    return f'{n:,}'


def markers(p):
    return (f'<defs>'
            f'<marker id="{p}-af" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="m-fwd"/></marker>'
            f'<marker id="{p}-ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="m-bwd"/></marker>'
            f'</defs>')


# ---------- vertical network (1(b) and 1(c)) ----------
def vnet(p, rows, aria):
    BX, BW, BH, GAP = 250, 260, 46, 46
    CX = BX + BW / 2
    pitch = BH + GAP
    top = 16
    height = top + len(rows) * pitch + 30
    s = [f'<svg viewBox="0 0 780 {height}" role="img" aria-label="{aria}">', markers(p)]
    first_lin_y = None
    for i, r in enumerate(rows):
        y = top + i * pitch
        cls = {'input': 'b-in', 'linear': 'b-lin', 'act': 'b-act', 'loss': 'b-loss'}[r['kind']]
        s.append(f'<rect x="{BX}" y="{y}" width="{BW}" height="{BH}" rx="6" class="{cls}"/>')
        s.append(f'<text x="{CX}" y="{y + 19}" text-anchor="middle" class="t-title">{r["title"]}</text>')
        s.append(f'<text x="{CX}" y="{y + 36}" text-anchor="middle" class="t-sub">{r["sub"]}</text>')
        if r.get('right1'):
            s.append(f'<text x="540" y="{y + 19}" class="t-param">{r["right1"]}</text>')
        if r.get('right2'):
            s.append(f'<text x="540" y="{y + 36}" class="t-calc">{r["right2"]}</text>')
        if r.get('left'):
            s.append(f'<text x="200" y="{y + 28}" text-anchor="end" class="t-grad">{r["left"]}</text>')
        if r['kind'] == 'linear' and first_lin_y is None:
            first_lin_y = y
        # forward arrow + shape label below the box
        ay1, ay2 = y + BH + 2, y + pitch - 3
        s.append(f'<line x1="{CX}" y1="{ay1}" x2="{CX}" y2="{ay2}" class="a-fwd" marker-end="url(#{p}-af)"/>')
        s.append(f'<text x="{CX + 12}" y="{(ay1 + ay2) / 2 + 4}" class="t-shape">{r["shape"]}</text>')
    # final output pill
    yl = top + len(rows) * pitch
    s.append(f'<text x="{CX}" y="{yl + 12}" text-anchor="middle" class="t-title">{rows[-1]["out"]}</text>')
    # backward arrow on the left, from the loss box up to the first Linear
    ly = top + (len(rows) - 1) * pitch + BH / 2
    s.append(f'<line x1="225" y1="{ly}" x2="225" y2="{first_lin_y + 4}" class="a-bwd" marker-end="url(#{p}-ab)"/>')
    s.append(f'<text x="40" y="{top + 12}" class="t-head">backward stores</text>')
    s.append('</svg>')
    return '\n'.join(s)


def mlp_rows(sizes, batch, input_sub):
    rows = [dict(kind='input', title='Input x', sub=input_sub, shape=f'({batch}, {sizes[0]})')]
    n_lin = len(sizes) - 1
    for i in range(n_lin):
        d, m = sizes[i], sizes[i + 1]
        last = i == n_lin - 1
        rows.append(dict(kind='linear', title=f'Linear {i + 1}', sub=f'{d} → {m}',
                         right1=f'{fmt(d * m + m)} parameters', right2=f'W {d}×{m} + b {m}',
                         left=f'dW{i + 1} ({d}, {m})  db{i + 1} ({m},)',
                         shape=f'({batch}, {m})' + ('  logits z' if last else '')))
        if not last:
            rows.append(dict(kind='act', title='ReLU', sub='max(0, x)', right1='0 parameters',
                             shape=f'({batch}, {m})'))
    rows.append(dict(kind='loss', title='Softmax + cross-entropy', sub='p = softmax(z),  L = −mean log p[y]',
                     right1='0 parameters', left=f'∂L/∂z = (p − onehot)/N', shape='',
                     out='L  (one number)'))
    return rows


# ---------- one Linear layer, neuron view ----------
def neuron_view():
    p = 'nv'
    xs_in, xs_out = 90, 360
    ys_in = [40, 95, 150, 205]
    ys_out = [67, 122, 177]
    s = [f'<svg viewBox="0 0 780 250" role="img" aria-label="A Linear layer with 4 inputs and 3 outputs: every input connects to every output, giving 12 weights and 3 biases.">']
    for i, yi in enumerate(ys_in):
        for j, yo in enumerate(ys_out):
            hl = (i == 1 and j == 0)
            s.append(f'<line x1="{xs_in + 16}" y1="{yi}" x2="{xs_out - 16}" y2="{yo}" class="{"w-hl" if hl else "w"}"/>')
    for i, yi in enumerate(ys_in):
        s.append(f'<circle cx="{xs_in}" cy="{yi}" r="16" class="n-in"/>')
        s.append(f'<text x="{xs_in}" y="{yi + 5}" text-anchor="middle" class="t-title">x{i + 1}</text>')
    for j, yo in enumerate(ys_out):
        s.append(f'<circle cx="{xs_out}" cy="{yo}" r="16" class="n-out"/>')
        s.append(f'<text x="{xs_out}" y="{yo + 5}" text-anchor="middle" class="t-title">y{j + 1}</text>')
        s.append(f'<text x="{xs_out + 24}" y="{yo + 5}" class="t-calc">+ b{j + 1}</text>')
    s.append(f'<text x="200" y="62" class="t-hl">W[2,1]</text>')
    s.append(f'<text x="{xs_in}" y="240" text-anchor="middle" class="t-head">D = 4 inputs</text>')
    s.append(f'<text x="{xs_out}" y="240" text-anchor="middle" class="t-head">M = 3 outputs</text>')
    s.append(f'<text x="470" y="70" class="t-param">y1 = x1·W[1,1] + x2·W[2,1]</text>')
    s.append(f'<text x="490" y="90" class="t-param">+ x3·W[3,1] + x4·W[4,1] + b1</text>')
    s.append(f'<text x="470" y="135" class="t-calc">weights: 4 × 3 = 12  (every line)</text>')
    s.append(f'<text x="470" y="155" class="t-calc">biases:  3          (one per output)</text>')
    s.append(f'<text x="470" y="182" class="t-param">parameters = D·M + M = 15</text>')
    s.append('</svg>')
    return '\n'.join(s)


# ---------- 10-layer chain (1(d)) ----------
SIG = [1.48e-04, 2.03e-04, 7.01e-04, 2.18e-03, 6.45e-03, 1.98e-02, 5.99e-02, 1.90e-01, 6.06e-01, 1.82e+00]
REL = [3.27, 1.40, 1.71, 2.14, 2.33, 2.86, 3.03, 2.87, 3.22, 3.25]


def short(v):
    if v >= 0.1:
        return f'{v:.2f}'
    m, e = f'{v:.1e}'.split('e')
    return f'{m}e{int(e)}'


def deep_chain():
    import math
    p = 'dc'
    X0, BW, PITCH, BY, BH = 118, 58, 88, 40, 46
    W = X0 + 10 * PITCH + 80
    s = [f'<svg viewBox="0 0 {W} 272" role="img" aria-label="A 10-layer MLP. Going backward, the sigmoid network\'s gradient norm falls from 1.8 at layer 10 to 0.00015 at layer 1, while the ReLU network stays between 1.4 and 3.3.">',
         markers(p)]
    s.append(f'<rect x="8" y="{BY}" width="84" height="{BH}" rx="6" class="b-in"/>')
    s.append(f'<text x="50" y="{BY + 19}" text-anchor="middle" class="t-title">Input x</text>')
    s.append(f'<text x="50" y="{BY + 36}" text-anchor="middle" class="t-sub">(512, 3072)</text>')
    for l in range(10):
        x = X0 + l * PITCH
        s.append(f'<line x1="{x - 24 if l else 94}" y1="{BY + BH / 2}" x2="{x - 3}" y2="{BY + BH / 2}" class="a-fwd" marker-end="url(#{p}-af)"/>')
        d = 3072 if l == 0 else 256
        m = 10 if l == 9 else 256
        s.append(f'<rect x="{x}" y="{BY}" width="{BW}" height="{BH}" rx="6" class="b-lin"/>')
        s.append(f'<text x="{x + BW / 2}" y="{BY + 19}" text-anchor="middle" class="t-title">L{l + 1}</text>')
        s.append(f'<text x="{x + BW / 2}" y="{BY + 36}" text-anchor="middle" class="t-tiny">{d}→{m}</text>')
        if l < 9:
            ax = x + BW + 15
            s.append(f'<text x="{ax}" y="{BY - 8}" text-anchor="middle" class="t-tiny">act</text>')
        # values
        cx = x + BW / 2
        op = 0.35 + 0.65 * (math.log10(SIG[l]) + 4) / 4.3
        s.append(f'<text x="{cx}" y="194" text-anchor="middle" class="t-sig" opacity="{max(0.5, min(1, op)):.2f}">{short(SIG[l])}</text>')
        s.append(f'<text x="{cx}" y="252" text-anchor="middle" class="t-relu">{short(REL[l])}</text>')
    xe = X0 + 10 * PITCH
    s.append(f'<line x1="{xe - 30}" y1="{BY + BH / 2}" x2="{xe - 3}" y2="{BY + BH / 2}" class="a-fwd" marker-end="url(#{p}-af)"/>')
    s.append(f'<rect x="{xe}" y="{BY}" width="60" height="{BH}" rx="6" class="b-loss"/>')
    s.append(f'<text x="{xe + 30}" y="{BY + 22}" text-anchor="middle" class="t-title">CE</text>')
    s.append(f'<text x="{xe + 30}" y="{BY + 40}" text-anchor="middle" class="t-tiny">→ loss</text>')
    # backward arrow
    s.append(f'<line x1="{xe + 30}" y1="{BY + BH + 30}" x2="{X0 + 2}" y2="{BY + BH + 30}" class="a-bwd" marker-end="url(#{p}-ab)"/>')
    s.append(f'<text x="{(xe + X0) / 2}" y="{BY + BH + 24}" text-anchor="middle" class="t-grad">backward: blame flows right → left, multiplied by each activation\'s slope</text>')
    s.append(f'<text x="8" y="172" class="t-head">Sigmoid: gradient norm of each layer, about 3 times smaller per step back</text>')
    s.append(f'<text x="8" y="230" class="t-head">ReLU: gradient norm of each layer, roughly level at every layer</text>')
    s.append('</svg>')
    return '\n'.join(s)


fig_b = vnet('fb', mlp_rows([20, 16, 16, 10], 8, 'random numbers, float64'),
             'Gradient-check MLP 20 to 16 to 16 to 10 with ReLU, batch of 8, 778 parameters in total.')
fig_c = vnet('fc', mlp_rows([3072, 512, 256, 10], 128, 'flattened 32×32×3 image'),
             'Training MLP 3072 to 512 to 256 to 10 with ReLU, batch of 128, 1,707,274 parameters in total.')


# ---------- standalone SVG files (fixed light palette so they read on any background) ----------
STYLE = '''<style>
.b-in{fill:#eef1f4;stroke:#96a1ae}
.b-lin{fill:#e7f0fb;stroke:#2a78d6;stroke-width:1.5}
.b-act{fill:#ffffff;stroke:#96a1ae;stroke-dasharray:4 3}
.b-loss{fill:#fbede6;stroke:#c4501f;stroke-width:1.5}
.a-fwd{stroke:#2a78d6;stroke-width:2}
.a-bwd{stroke:#c4501f;stroke-width:2.5}
.m-fwd{fill:#2a78d6}.m-bwd{fill:#c4501f}
text{font-family:"Segoe UI",Helvetica,Arial,sans-serif}
.t-title{fill:#1a1e24;font-size:13px;font-weight:700}
.t-sub,.t-calc,.t-tiny,.t-shape,.t-param,.t-grad,.t-hl,.t-sig,.t-relu{font-family:"DejaVu Sans Mono",Menlo,Consolas,monospace}
.t-sub,.t-calc{fill:#5a6470;font-size:12px}
.t-tiny{fill:#5a6470;font-size:10.5px}
.t-shape{fill:#2a78d6;font-size:12px;font-weight:600}
.t-param{fill:#1a1e24;font-size:12.5px;font-weight:600}
.t-grad,.t-hl,.t-sig{fill:#c4501f;font-size:11.5px;font-weight:600}
.t-relu{fill:#2a78d6;font-size:12px;font-weight:600}
.t-head{fill:#5a6470;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.w{stroke:#96a1ae;stroke-width:1}.w-hl{stroke:#c4501f;stroke-width:2.5}
.n-in{fill:#eef1f4;stroke:#96a1ae}.n-out{fill:#e7f0fb;stroke:#2a78d6;stroke-width:1.5}
</style>'''


def standalone(svg):
    head, rest = svg.split('>', 1)
    w, h = head.split('viewBox="0 0 ')[1].split('"')[0].split()
    head = head.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return f'{head}>{STYLE}<rect width="{w}" height="{h}" rx="10" fill="#ffffff"/>{rest}'


FIGURES = {
    'fig0_linear_layer.svg': neuron_view(),
    'fig1_1b_gradient_check_mlp.svg': fig_b,
    'fig2_1c_training_mlp.svg': fig_c,
    'fig3_1d_ten_layer_gradients.svg': deep_chain(),
}
for name, svg in FIGURES.items():
    with open(os.path.join(OUT_DIR, name), 'w') as f:
        f.write(standalone(svg) + '\n')
    print('wrote', name)
