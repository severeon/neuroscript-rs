# Shape Algebra: Catalog of Fundamental Operators

Below is a rigorously structured list.

Each group includes:

* **Operator name**
* **Symbol / keyword suggestion**
* **Short definition**
* **Examples**

This is all shape-level; no tensors.

---

# A. **Cardinality Operators** (total size)

### 1. `size`

**Keyword:** `size(x)`
**Meaning:** Product of dimensions.

```
size([4,4,8]) = 128
```

### 2. `fits`

**Keyword:** `A fits B` or `fits(A, B)`
**Meaning:** `size(A)` ≤ `size(B)`

```
[64] fits [256]     → true
```

### 3. `divides`

**Keyword:** `A | B`
**Meaning:** `size(B) % size(A) == 0`

```
[64] | [256]        → true
```

### 4. `quotient`, `remainder`

```
q = quotient([192], [256]) → 1  
r = remainder([192], [256]) → 64
```

### 5. `gcd_size`, `lcm_size`

Useful for alignment and SIMD rules.

---

# B. **Axiswise Operators** (per-dimension)

These interpret shapes as ℕᵈ vectors.

Let A = `[a1, a2, ..., an]` and B = `[b1, b2, ..., bn]`.

### 1. `==`, `<=`, `<`, `>=`, `>`

Axiswise boolean checks.

```
[32,16] <= [32,32]    → true
```

### 2. `|` (axiswise divisibility)

```
[4, 8] | [16, 64]     → true
```

### 3. `gcd`, `lcm` at each axis

```
gcd([6,4], [9,8]) = [3,4]
```

### 4. `min`, `max`

Shapes as a lattice.

```
max([2,5], [4,3]) = [4,5]
```

---

# C. **Broadcasting Relations**

### 1. `broadcastable(A, B)`

General operator.

```
broadcastable([4,1,8], [4,10,8]) → true
broadcastable([4,3,1], [4,10,8]) → false
```

### 2. `weak_axis(i)`

A predicate indicating that dimension `i == 1`.

```
weak_axis(1) for shape [32,1,256] → true
```

### 3. Pattern syntax for broadcast:

```
[*, 1, *] ⟷ broadcastable_on_axis(1)
```

---

# D. **Factorization & Reshape Algebra**

### 1. `factor(n)`

Returns allowed decompositions of a dimension.

```
factor(64) = [ [8,8], [4,4,4], [2,2,2,2,2,2] ... ]
```

### 2. `reshapeable(A, B)`

Check if products match.

```
reshapeable([4,4,8], [2,2,2,4,4]) → true
```

### 3. `refine_axis(i, factors)`

Split axis i into new axes.

```
[64] refine_axis 0 by [8,8] → [8,8]
```

### 4. `coarsen_axes(i,j)`

Merge two axes.

```
[8,8] coarsen → [64]
```

---

# E. **Containment, Packing, and Tiling**

### 1. `≤` (containment)

Axiswise containment.

```
[8,4] ≤ [8,16] → true
```

### 2. `tiles(A, B)`

A tiles B if all axes divide:

```
tiles([2,2], [8,6]) → false
tiles([2,3], [8,6]) → true
```

### 3. `tile_count(A, B)`

```
tile_count([2,3], [8,6]) = [4,2]
```

### 4. `aligned(A, k)`

All axes multiples of k.

```
aligned([64,128], 16) → true
```

---

# F. **Topological Relations**

(non-numerical structure)

### 1. `permutes(A, B)`

They have same dims, order can differ.

```
permutes([2,3,4], [3,2,4]) → true
```

### 2. `rank(A)`

Number of dims.

### 3. `same_cardinality(A, B)`

`size(A) == size(B)`.

### 4. `flatten(A)`

Return 1D `[size(A)]`

---

# G. **Logical Combinators**

For guard clauses.

* `and`
* `or`
* `not`
* `where`
* `exists`
* `for all` (axis quantifiers)

---

# H. **Shape Pattern Language (Prolog-like)**

### Wildcards

* `*` → match any axis, bind
* `_` → match any axis, ignore
* `$name` → capture binding
* Literal integers → enforce match

### Examples:

### **1. Match a broadcastable middle axis**

```
match [A, 1, C]:
    broadcast_axis = 1
```

### **2. Match any shape divisible by 64**

```
match [X] where X % 64 == 0:
    ok
```

### **3. Match tileable 2D shape**

```
match [H, W] where H % 8 == 0 and W % 8 == 0:
    tile_by = 8
```

### **4. Match two shapes that can be broadcast together**

```
match (A, B) where broadcastable(A, B):
    allowed
```

### **5. Match shape with square spatial dimensions**

```
match [B, H, H, C]:
    square = true
```

### **6. Match shape where remainder(A,B) is power-of-two**

```
match (A, B) 
    where rem(size(B), size(A)) in powers_of_two:
        ok
```

---

# 3. Summary of All Operator Names

### **Arithmetic**

* `size`
* `fits`
* `divides`
* `quotient`
* `remainder`
* `gcd_size`, `lcm_size`

### **Axiswise**

* `==, <=, >=, |, gcd, lcm, min, max`

### **Broadcast**

* `broadcastable`
* `weak_axis`
* broadcast patterns with `1` literal

### **Factorization**

* `factor`
* `reshapeable`
* `refine_axis`
* `coarsen_axes`

### **Containment & Tiling**

* `tiles`
* `tile_count`
* `aligned`
* axiswise `≤`

### **Topology**

* `permutes`
* `rank`
* `same_cardinality`
* `flatten`

### **Logic**

* `where`
* `and`, `or`, `not`
* `exists axis`
* `forall axis`

### **Pattern**

* `*`, `_`, `$name`, integers
