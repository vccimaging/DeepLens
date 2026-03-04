# Aspheric Lens Design Principles: Surface Selection, Placement, and Polynomial Order

**Date**: 2026-02-27
**Context**: DeepLens project — guiding aspheric surface allocation in differentiable lens design

---

## 1. Core Question

In a multi-element optical system using both spherical and aspheric surfaces:
1. **How many** elements should be aspheric?
2. **Which** element(s) should be aspheric?
3. **What polynomial order** (number of aspheric coefficients) should be used?

---

## 2. Fundamental Principle: Asphere Placement Relative to Aperture Stop

This is the single most important rule in aspheric lens design, derived from **Seidel aberration theory and stop-shift equations**:

### 2.1 Asphere Near the Aperture Stop → Corrects Spherical Aberration Only

> "When the stop is at the aspheric surface, only spherical aberration is contributed, given that all the beams see the same portion of the surface."
> — Prof. Jose Sasian, OPTI 517, University of Arizona

- An aspheric surface **at or near the aperture stop** affects **only spherical aberration** (SA)
- It has **no effect** on coma, astigmatism, field curvature, or distortion
- **Classic example**: The **Schmidt camera** places an aspheric corrector plate at the stop to correct SA of a spherical mirror without introducing field aberrations

### 2.2 Asphere Away from the Stop → Corrects Field-Dependent Aberrations

> "When the stop is away from the surface, different field beams pass through different parts of the aspheric surface and other aberrations are contributed."

- An aspheric surface **far from the stop** can correct **coma, astigmatism, distortion**
- Off-axis beams sample different zones of the asphere, creating field-dependent corrections
- **Example**: Field-flatteners placed near the image plane

### 2.3 The Two-Asphere Strategy

A practical and widely used approach:

| Asphere | Position | Corrects |
|---------|----------|----------|
| Asphere 1 | Near aperture stop | Spherical aberration |
| Asphere 2 | Away from stop (near field) | Coma, astigmatism, distortion |

This two-asphere strategy often provides sufficient aberration control for many optical systems.

---

## 3. How Many Aspheric Surfaces to Use

### 3.1 General Principle: Minimum Necessary

> "A single aspheric surface in a system is often sufficient, and it is usually neither necessary nor advisable to use aspheric optics throughout a system."
> — RP Photonics

**Guidelines by system complexity:**

| System Type | Typical # Aspheres | Notes |
|------------|-------------------|-------|
| Triplet (e.g., Cooke triplet) | 1 | Single asphere on first surface can give up to 4x resolution improvement |
| Double Gauss | 1–2 | Two aspheric surfaces on one element can reach diffraction-limited performance |
| Zoom lens (10+ elements) | 2–3 | Two aspheres can replace multiple spherical elements |
| Smartphone camera (5–8 elements) | All surfaces | Compact form factor requires heavy aspherization |
| Schmidt-Cassegrain | 1 | Aspheric corrector plate at stop |

### 3.2 Diminishing Returns

- Adding the first asphere typically gives the **largest performance gain**
- Each additional asphere provides **diminishing improvement**
- Multiple aspheres on pupil-conjugate surfaces can **cancel each other** ("duel"), producing large opposing aberrations — a single asphere with the net correction is often better

### 3.3 Decision Factors

1. **Performance requirement**: Is the system diffraction-limited or visual quality?
2. **Size/weight constraints**: Fewer elements with aspheres vs. more spherical elements
3. **Manufacturing budget**: Aspheres are ~6x more expensive than spherical equivalents
4. **Production volume**: High volume → molded aspheres are cost-effective; low volume → spherical preferred

---

## 4. Which Element Should Be Aspheric

### 4.1 Systematic Approach (Zemax "Find Best Asphere" Methodology)

OpticStudio's "Find Best Asphere" tool evaluates each surface as a candidate:

**Candidate requirements:**
- Surface must be of type Standard (not already aspheric)
- No existing conic value
- Must define a boundary between **air and glass** (cemented surfaces usually make poor aspheres)
- Curvature must be variable or controlled by a solve

**Workflow:**
1. Convert each candidate surface to aspheric type
2. Set aspheric terms as variables
3. Run local optimization (damped least squares)
4. Compare merit function values
5. Select the surface giving the **largest merit function improvement**

### 4.2 Physical Intuition for Surface Selection

| Criterion | Preferred Surface |
|-----------|-------------------|
| Correct spherical aberration | Surface near aperture stop |
| Correct field curvature | Surface near image plane |
| Correct coma/astigmatism | Surface away from stop, in diverging/converging beam |
| Maximum ray height | Surface where marginal ray is tallest (most SA contribution) |
| Air-glass interface | Prefer over cemented (glass-glass) interfaces |
| Minimize aspheric departure | Surface with largest base curvature (aspheric terms are smaller corrections) |
| **Avoid outermost surfaces** | **Inner surfaces preferred over front/back element surfaces** |

### 4.3 Avoid Outermost Surfaces (Front and Back Elements)

In multi-element camera lens design, the **first surface** (front of the first element) and the **last surface** (back of the last element) should generally **not** be aspherized. This is a well-known practical rule:

1. **Large diameter**: The front element is typically the largest surface in the system. Aspherizing large surfaces is significantly more expensive and difficult to manufacture and test.
2. **Protective role**: The front element often serves as a protective meniscus (e.g., in Double Gauss designs). Its shape is constrained by mechanical and environmental requirements, not aberration correction.
3. **Low aberration leverage**: The front surface is far from the aperture stop and sees nearly identical ray bundles for all field angles — it has limited ability to correct field-dependent aberrations. The back surface is close to the image plane where ray heights are small, giving aspheric terms little leverage.
4. **Environmental exposure**: The front surface is exposed to dust, moisture, and handling. Complex aspheric profiles on the outermost surface are more vulnerable to damage and harder to clean/recoat.
5. **Better inner candidates exist**: Inner surfaces (closer to the aperture stop or in converging/diverging beams) have higher marginal ray heights and stronger aberration contributions, making them more effective aspheric candidates.

> **Rule**: When selecting subsequent aspheric surfaces (after the first near-stop asphere), **exclude the first and last refractive surfaces** from the candidate pool. Choose inner surfaces that are far from the stop but not at the system boundary.

**Exceptions** (rare):
- **Smartphone camera lenses**: All surfaces including outermost are aspherized because the system uses injection-molded plastic elements where aspherization adds no manufacturing cost.
- **Single-element systems**: The front surface may be the only candidate (e.g., aspheric singlet condenser).

### 4.4 Bending-First Strategy

From aberration theory (Sasian, Pulsar DevOptical):
1. **First**: Optimize lens bending (base curvatures) to minimize coma
2. **Then**: Apply aspheric terms to correct remaining spherical aberration
3. The required aspheric coefficient becomes **deterministic** once bending is fixed

---

## 5. Aspheric Polynomial Order Selection

### 5.1 The Even Asphere Equation

The standard even aspheric surface sag:

```
z(r) = (c * r^2) / (1 + sqrt(1 - (1+k)*c^2*r^2)) + α4*r^4 + α6*r^6 + α8*r^8 + α10*r^10 + ...
```

Where:
- `c` = base curvature (1/radius)
- `k` = conic constant
- `α_i` = aspheric coefficients (even orders only for rotationally symmetric)

### 5.2 What Each Order Corrects

| Coefficient | Power | Primary Effect |
|------------|-------|----------------|
| k (conic) | r^4 effective | Third-order spherical aberration |
| α4 | r^4 | Third-order spherical (redundant with conic if both used) |
| α6 | r^6 | Fifth-order spherical aberration |
| α8 | r^8 | Seventh-order spherical aberration |
| α10 | r^10 | Ninth-order spherical aberration |
| α12+ | r^12+ | Higher-order residuals |

**Key insight**: Higher-order terms affect **mostly the outer zones** of the aperture due to the high powers of r.

### 5.3 Order Selection Guidelines

#### Rule: Start Low, Add Incrementally

> "Start with α4, then toggle α6 if it's not enough, and so on."
> — DevOptical Part 22

**Practical workflow:**
1. Start with **conic constant only** (k) — often sufficient for moderate systems
2. Add **α4** if conic alone is insufficient (but do NOT use both k and α4 for the same correction — they compete)
3. Add **α6** for fast systems (low f/#) or wide-field systems
4. Add **α8, α10** only if residual higher-order aberrations persist
5. **Stop** when additional terms provide negligible merit function improvement

#### Guidelines by System Speed

| System f/# | Typical Max Order | Reasoning |
|-----------|------------------|-----------|
| f/4 or slower | Conic only or α4 | Third-order SA dominates |
| f/2 – f/4 | Up to α6 or α8 | Fifth-order terms become significant |
| f/1.4 – f/2 | Up to α8 or α10 | Higher-order correction needed |
| < f/1.4 (smartphone) | α10 – α16+ | Extreme asphericity required |

#### When to Use Conic vs. Power Series

| Approach | Best For | Avoid When |
|----------|----------|------------|
| Conic constant only | Simple systems, conics near-optimal | Higher-order correction needed |
| Power series (α4, α6, ...) | Higher-order correction, complex profiles | Only third-order correction needed |
| Conic + power series | Maximum flexibility | Risk of coefficient competition at 4th order |

#### Why Conic (k) and α₄ Compete — Taylor Expansion Analysis

The conic base sag, when Taylor-expanded around r=0, reveals which parameters contribute to which power of r:

```
z(r) = c·r² / (1 + √(1-(1+k)·c²·r²))  +  α₂·r²  +  α₄·r⁴  +  α₆·r⁶  + ...
        \__________ conic base _________/   \_______ polynomial terms ______/

Taylor expansion of conic base:
z(r) ≈ (c/2)·r²  +  ((1+k)·c³/8)·r⁴  +  ((1+k)²·c⁵/16)·r⁶  + ...
        ^^^^^^^^      ^^^^^^^^^^^^^^^^
        r² term        r⁴ term (depends on k!)
```

| Power | Contributors | Competition |
|-------|-------------|-------------|
| r² | Base curvature `c` **and** `α₂` | α₂ competes with c → **α₂ is always set to 0** |
| r⁴ | Conic constant `k` (via `(1+k)c³/8`) **and** `α₄` | **k competes with α₄** → don't vary both simultaneously |
| r⁶ | Conic `k` (higher-order expansion) **and** `α₆` | Weaker competition |

**Key takeaway**: The conic constant `k` does NOT compete with the 2nd-order term α₂ (a common misconception). Rather:
- **α₂ competes with base curvature c** (both are r² terms) — this is why α₂ is always zero
- **Conic k competes with α₄** (both contribute r⁴ terms) — varying both simultaneously creates a degenerate parameter space and causes optimizer instability

**In practice**, choose one of:
- **Conic k alone** (k≠0, α₄=0): for simple third-order SA correction (parabola, hyperbola, ellipse)
- **α₄ alone** (k=0, α₄≠0): if you plan to add higher-order terms α₆, α₈... for a unified polynomial representation
- **Never vary both k and α₄** simultaneously during optimization

### 5.4 Which Surface to Increase Aspheric Order On — Selection Principles

When the current aspheric orders are insufficient and you need to add higher-order terms, **which surface should get the next term?** The following principles guide this decision.

#### Principle 1: Prioritize the Surface with the Largest Marginal Ray Height

The aspheric correction to spherical aberration from a coefficient αₙ scales as **yⁿ** (the n-th power of the marginal ray height at that surface). This means:

- The surface where the **beam is widest** (largest marginal ray height y) gets the most correction per unit of aspheric coefficient
- A smaller αₙ value is needed → the surface stays closer to spherical → **easier to manufacture and test**
- Conversely, adding high-order terms to a surface with a small beam gives negligible effect

> **Rule**: Increase aspheric order first on the surface with the **tallest marginal ray**.

#### Principle 2: Prioritize the Surface with the Largest Seidel Aberration Contribution

Use the **Seidel surface contribution analysis** (available in Zemax, CODE V, etc.) to identify which surface contributes the most to the dominant residual aberration:

- If surface j has the largest spherical aberration contribution (S_I) → increasing its aspheric order gives the most leverage
- The aspheric cap correction δS_I is proportional to Δn · y⁴ · (asphericity), where Δn is the refractive index change at the interface
- **Higher Δn** (e.g., glass-to-air, n=1.5→1.0) amplifies the aspheric effect compared to low-contrast interfaces

> **Rule**: Aspherize the surface that **contributes the most aberration** — it has the most to gain.

#### Principle 3: Match the Term Order to the Dominant Residual Aberration

After optimization with current orders, analyze the **residual wavefront error** (e.g., via Zernike decomposition or wavefront map):

| Residual Pattern | Likely Cause | Where to Add Terms |
|---|---|---|
| On-axis rotationally symmetric residual | Higher-order spherical (W060, W080) | Surface **near the aperture stop** (where ȳ≈0, so only SA is affected) |
| Field-dependent blur (varies across FOV) | Higher-order coma, astigmatism | Surface **away from the stop** (where ȳ/y is large) |
| Residual only at edge of aperture | Very high-order terms (r⁸, r¹⁰...) | Surface with the **largest clear aperture** |
| Residual only at edge of field | Field curvature, oblique SA | Surface **near the image plane** or field lens |

> **Rule**: The **spatial pattern** of the residual tells you both **which order** and **which surface** to target.

#### Principle 4: Higher-Order Terms Affect Outer Zones Most

Since α₆·r⁶, α₈·r⁸, α₁₀·r¹⁰ grow very rapidly with r:

- They primarily shape the **outer annular zones** of the aperture
- On a surface with a small beam, high-order terms are negligible (r is small, so r⁸ ≈ 0)
- On a surface with a large beam, high-order terms have real leverage

> **Rule**: Only add high-order terms to surfaces where the **clear aperture is large enough** for those terms to matter.

#### Principle 5: One Surface, One Term at a Time

Avoid increasing order on multiple surfaces simultaneously:

1. Add **one** new term (e.g., α₈) to the **best candidate** surface
2. **Re-optimize** all existing parameters (curvatures, spacings, existing aspheric coefficients)
3. Evaluate the **merit function improvement**
4. If improvement is significant, keep it; if marginal, remove it
5. **Then** consider adding a term to a different surface or the next higher order

This avoids the "dueling aspheres" problem where two surfaces develop large, opposing aspheric departures.

#### Principle 6: For Pupil-Conjugate Surfaces, Choose Only One

If two surfaces are conjugate to the same pupil (e.g., both near the aperture stop), their aspheric terms correct the **same aberrations** and will compete. Increasing order on both simultaneously:

- Creates a degenerate optimization space
- Produces large opposing coefficients that are sensitive to manufacturing errors
- **Choose the one** with the larger marginal ray height or lower manufacturing difficulty

#### Summary: Decision Flowchart for Increasing Aspheric Order

```
Residual aberration too large after optimization?
│
├── Identify dominant residual (wavefront/Zernike analysis)
│   ├── On-axis spherical? → Target surface NEAR STOP
│   └── Field-dependent?   → Target surface AWAY FROM STOP
│
├── Among candidate surfaces, rank by:
│   1. Largest marginal ray height (y)
│   2. Largest Seidel aberration contribution
│   3. Largest clear aperture (for high-order terms)
│   4. Highest Δn at the interface
│
├── Add ONE term to the top-ranked surface
├── Re-optimize all parameters
├── Evaluate improvement
│   ├── Significant → Keep, consider next surface
│   └── Marginal   → Remove, system is at its limit
│
└── Repeat until performance target met or diminishing returns
```

### 5.5 Manufacturing Constraints on Order

- **Higher-order terms** → steeper local slopes → harder to manufacture and test
- **Slope of aspheric departure** has greater manufacturing impact than the amplitude
- **Inflection points** (slope reversals) in the aspheric profile drastically increase manufacturing cost
- Orthogonal representations (Qcon, Qbfs from Forbes) minimize coefficient interaction and improve manufacturability assessment

---

## 6. Relevance to Differentiable Lens Design (DeepLens)

Recent work in differentiable optics addresses these questions computationally:

### 6.1 Current Approaches

- **DeepLens / dO framework**: Differentiable ray tracing tracks gradients of all optical parameters (curvature, position, conic, polynomial coefficients) — enabling gradient-based optimization of aspheric terms
- **Curriculum learning** (Yang et al., Nature Communications 2024): Ab initio design starting from random surfaces, automatically learning highly aspheric profiles without human-specified initial shapes
- **End-to-end optimization**: Joint optimization of lens surfaces and image reconstruction networks

### 6.2 Open Challenges for Automated Aspherization

1. **Discrete decisions**: Which surfaces to make aspheric is a combinatorial/discrete choice — not directly handled by gradient descent
2. **Order selection**: Number of polynomial terms is a hyperparameter, not typically optimized
3. **Manufacturability constraints**: Gradient-based optimizers may produce un-manufacturable profiles unless constrained
4. **Local minima**: Multiple aspheres can create competing solutions (the "dueling" problem)

### 6.3 Potential Research Directions

- **Learnable aspherization**: Use differentiable relaxation to jointly learn which surfaces should be aspheric (e.g., Gumbel-Softmax for discrete surface type selection)
- **Adaptive polynomial order**: Start with low-order and progressively add terms during optimization (curriculum approach)
- **Manufacturability-aware loss**: Penalize slope discontinuities, inflection points, and extreme aspheric departures
- **Regularization**: L1/L2 on aspheric coefficients to encourage sparsity (fewer active terms)

---

## 7. Summary of Design Rules

### Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              ASPHERIC DESIGN DECISION TREE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. HOW MANY ASPHERES?                                      │
│     ├── Start with ONE (biggest bang for buck)               │
│     ├── Add second only if first is insufficient             │
│     └── Avoid aspheres on pupil-conjugate surfaces           │
│                                                             │
│  2. WHICH SURFACE?                                          │
│     ├── Near stop → spherical aberration                     │
│     ├── Away from stop → field aberrations                   │
│     ├── Prefer air-glass over cemented interfaces            │
│     ├── Choose surface with highest marginal ray             │
│     └── AVOID outermost surfaces (front/back elements)       │
│                                                             │
│  3. WHAT ORDER?                                             │
│     ├── Start: conic constant OR α4 (not both)              │
│     ├── Add α6 for fast systems (< f/2.8)                   │
│     ├── Add α8, α10 for very fast (< f/1.8)                │
│     ├── Stop when improvement < threshold                    │
│     └── Check for inflection points (manufacturability)      │
│                                                             │
│  4. WHERE TO INCREASE ORDER?                                 │
│     ├── Surface with largest marginal ray height (y)         │
│     ├── Surface with largest Seidel aberration contribution  │
│     ├── Near stop → for on-axis spherical residual           │
│     ├── Away from stop → for field-dependent residual        │
│     ├── Surface with largest clear aperture (for α8+)       │
│     └── Higher Δn at interface → more aspheric leverage      │
│                                                             │
│  5. OPTIMIZATION WORKFLOW                                    │
│     ├── Optimize bending first (spherical surfaces)          │
│     ├── Then add asphericity to best candidate               │
│     ├── Add ONE term to ONE surface at a time                │
│     ├── Re-optimize all parameters after each addition       │
│     └── Never increase order on pupil-conjugate pair         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key References

### Textbooks & Courses
- Prof. Jose Sasian, **OPTI 517 Lens Design**, University of Arizona — [Aberrations Lecture](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/L3_OPTI517_Aberrations.pdf), [Spherical Aberration](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/L8_OPTI517_Spherical_Aberration.pdf)
- W. Smith, *Modern Optical Engineering*, McGraw-Hill

### Design Guides
- [RP Photonics — Aspheric Optics](https://www.rp-photonics.com/aspheric_optics.html)
- [Edmund Optics — All About Aspheric Lenses](https://www.edmundoptics.com/knowledge-center/application-notes/optics/all-about-aspheric-lenses/)
- [Edmund Optics — Shape Factor Influence in Aspheric Lens Design](https://www.edmundoptics.com/knowledge-center/application-notes/optics/shape-factor-influence-in-aspheric-lens-design/)
- [DevOptical Part 22: Aspheric Lenses](https://www.thepulsar.be/article/-devoptical-part-22--aspheric-lenses)
- [Zemax — How to Use the Find Best Asphere Tool](https://support.zemax.com/hc/en-us/articles/1500005489261-How-to-use-the-find-best-asphere-tool)
- [Zemax — Aspheric Surfaces Part 1](https://support.zemax.com/hc/en-us/articles/31584981420691-Aspheric-Surfaces-Part-1-Introduction-to-Aspherical-Surfaces-in-Optical-Design)

### Manufacturing
- [Tech Briefs — Properly Designing and Specifying Aspheric Lenses](https://www.techbriefs.com/component/content/article/36881-properly-designing-and-specifying-aspheric-lenses)
- [Designing and Specifying Aspheres for Manufacturability (U. Arizona)](https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Gerard-Desroches.pdf)
- [Optica — Manufacturability Estimates for Optical Aspheres](https://opg.optica.org/oe/fulltext.cfm?uri=oe-19-10-9923&id=213662)

### Differentiable Lens Design
- Yang et al., "Curriculum Learning for ab initio Deep Learned Refractive Optics", *Nature Communications* 2024 — [Paper](https://www.nature.com/articles/s41467-024-50835-7)
- Wang et al., "dO: A Differentiable Engine for Deep Lens Design", 2022 — [Project](https://vccimaging.org/Publications/Wang2022DiffOptics/)
- Sun et al., "End-to-End Hybrid Refractive-Diffractive Lens Design", *SIGGRAPH Asia* 2024 — [Paper](https://dl.acm.org/doi/10.1145/3680528.3687640)

---

## 9. Implications for DeepLens Development

For the DeepLens differentiable lens design framework, these principles suggest:

1. **Parameterization**: The current even-asphere polynomial representation is standard and appropriate
2. **Initialization**: Start optimization with all-spherical surfaces, then progressively enable aspheric terms (matches curriculum learning philosophy)
3. **Regularization**: Add penalties for:
   - Large aspheric departures from best-fit sphere
   - Slope discontinuities and inflection points
   - High-order coefficient magnitudes
4. **Architecture search**: The discrete question of "which surfaces are aspheric" could be formulated as a differentiable architecture search problem (similar to NAS in deep learning)
5. **Tolerance**: Aspheric surfaces are more sensitive to manufacturing/alignment tolerances than spherical — tolerance analysis should weight aspheric surfaces more heavily
