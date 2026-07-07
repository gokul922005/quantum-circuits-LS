"""Decompose an arbitrary unitary into the universal basis {H, T, CNOT}.

A unitary is
lowered in stages:

    1. twolevel_decomposition:  Unitary     -> list[TwoLevel]
    2. decompose_twolevel:      TwoLevel    -> SingleQubitGate + ControlledU
    3. decompose_controlledU:   ControlledU -> CU + CNOT
    4. decompose_cu:            CU          -> SingleQubitGate + CNOT
    5. decompose_to_ht:         SingleQubitGate -> H / T words (using rotation.py)

Numpy types: a `Unitary` (N x N) and a 2x2 gate block are
both np.ndarray (complex128); a `ComplexVec` is a 1-D np.ndarray. A `Circuit` is a
Python list of gate objects, each exposing `to_unitary()`; gates are stored in
order of application (the first gate is applied first, i.e. the rightmost matrix
factor).

Every function/method below is a stub for you to implement; See "03 - Completing the Decomposition.pdf" for the recommended order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
 
import numpy as np

import rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    # TODO: implement.
    """Return n such that N = 2^n."""

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("Matrix dimension must be a positive power of 2.")

    return int(np.log2(N))

# ---------------------------------------------------------------------------
# Gate representations
#
# Each is a sparse description of an operation with a `to_unitary()` returning the
# full N x N matrix. As the decomposition progresses, gates get rewritten into
# simpler ones. The 2x2 block `unitary` is a (2, 2) complex ndarray.
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand to the full `size` x `size` matrix: identity except the 2x2 block
        placed at rows/cols (level0, level1).
        """
        # TODO: implement.
        """Expand to the full size × size matrix."""

        U = np.eye(self.size, dtype=rotation.DTYPE)

        i = self.level0
        j = self.level1

        U[i, i] = self.unitary[0, 0]
        U[i, j] = self.unitary[0, 1]
        U[j, i] = self.unitary[1, 0]
        U[j, j] = self.unitary[1, 1]

        return U

@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand the 2x2 to N dimensions: for each basis index whose `qubit` bit is
        0, fill the 2x2 block linking it to its partner (that bit = 1).
        """
        # TODO: implement.
        """Expand the 2x2 gate to the full n-qubit unitary."""

        N = 1 << self.n
        U = np.eye(N, dtype=rotation.DTYPE)

        mask = 1 << self.qubit

        for i in range(N):
            # Only process states where the target bit is 0
            if (i & mask) == 0:
                j = i | mask

                U[i, i] = self.unitary[0, 0]
                U[i, j] = self.unitary[0, 1]
                U[j, i] = self.unitary[1, 0]
                U[j, j] = self.unitary[1, 1]

        return U

@dataclass
class ControlledU:
    """A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity everywhere except the single controlled block: the pair (all
        ones except the target bit, all ones).
        """
        # TODO: implement.
        N = 1 << self.n
        U = np.eye(N, dtype=rotation.DTYPE)

        base = 0
        for q in range(self.n):
            if q != self.target:
                base |= (1 << q)

        i = base
        j = base | (1 << self.target)

        U[i, i] = self.unitary[0, 0]
        U[i, j] = self.unitary[0, 1]
        U[j, i] = self.unitary[1, 0]
        U[j, j] = self.unitary[1, 1]

        return U


@dataclass
class CU:
    """A singly-controlled single-qubit gate C(U): apply the 2x2 `unitary` to
    `target` iff `control` is 1. The full U(2) (global phase kept) is stored, since
    under a control the global phase becomes a physical relative phase. This is the
    recursion leaf of decompose_controlled.
    """

    n: int
    control: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where `unitary` acts on `target`."""
        # TODO: implement.
        
        N = 1 << self.n
        U = np.eye(N, dtype=rotation.DTYPE)

        control_mask = 1 << self.control
        target_mask = 1 << self.target

        for i in range(N):
            # control must be 1 and target must be 0
            if (i & control_mask) and not (i & target_mask):
                j = i | target_mask

                U[i, i] = self.unitary[0, 0]
                U[i, j] = self.unitary[0, 1]
                U[j, i] = self.unitary[1, 0]
                U[j, j] = self.unitary[1, 1]

        return U


@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1. Its 2x2 is fixed to
    Pauli-X, so (unlike CU) it stores no unitary.
    """

    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where X swaps the target's 0/1
        amplitudes.
        """
        # TODO: implement.
        N = 1 << self.n
        U = np.eye(N, dtype=rotation.DTYPE)

        control_mask = 1 << self.control
        target_mask = 1 << self.target

        for i in range(N):
            # control = 1 and target = 0
            if (i & control_mask) and not (i & target_mask):
                j = i | target_mask

                # Replace the 2×2 block by Pauli-X
                U[i, i] = 0
                U[i, j] = 1
                U[j, i] = 1
                U[j, j] = 0

        return U


@dataclass
class Swap:
    """A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`. `control_vals` has size n and is
    indexed by qubit; control_vals[target] is unused.
    """

    target: int
    control_vals: list[bool]


# A gate is any of the sparse representations above; a circuit is a list of gates.
Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list  # list[Gate]
TwoLevels = list  # list[TwoLevel]


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    """Full N x N unitary of a whole circuit. Gates are stored in order of
    application, so the product premultiplies (first gate is the rightmost factor):
    result = g_last @ ... @ g_1. Assumes the circuit is non-empty.
    """
    # TODO: implement.
    if len(circuit) == 0:
        raise ValueError("Circuit must not be empty.")

    N = circuit[0].to_unitary().shape[0]
    U = np.eye(N, dtype=rotation.DTYPE)

    for gate in circuit:
        U = gate.to_unitary() @ U

    return U


def to_circuit(two_levels: TwoLevels) -> Circuit:
    """Wrap a two-level sequence as a circuit, so decompose_unitary /
    twolevel_decomposition output flows straight into a Circuit.
    """
    # TODO: implement.
    return list(two_levels)


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Elementwise difference between two same-size unitaries, ignoring an overall
    global phase: align b to a by the phase of their Hilbert-Schmidt overlap
    <b, a> = sum conj(b_ij) a_ij, then compare. ~0 means equal up to global phase.
    """
    # TODO: implement.
    overlap = np.vdot(b, a)

    if np.isclose(overlap, 0):
        phase = 1.0
    else:
        phase = overlap / abs(overlap)

    b_aligned = phase * b

    return np.linalg.norm(a - b_aligned)

# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries (see cpp/src/TwoLevel.h)
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    """The 2x2 unitary [[conj(x), conj(y)], [-y, x]] / norm. Premultiplying it onto
    a column with entries (x, y) at two levels rotates the amplitude at the second
    level onto the first, leaving the real `norm` there and 0 below.
    """
    # TODO: implement.
    return np.array([
        [np.conj(x), np.conj(y)],
        [-y,         x]
    ], dtype=rotation.DTYPE) / norm


def decompose_vector(vec: np.ndarray) -> TwoLevels:
    """Given the first column of a unitary, return a sequence of two-levels which,
    when premultiplied onto the unitary, make its first column be (1, 0, 0, ...).
    Walk from the bottom up, using `align` at each pivot to zero out one entry; the
    running pivot holds the accumulated real norm after the first rotation.
    """
    # TODO: implement.
    vec = vec.copy()

    N = len(vec)

    result = []

    for i in range(N - 1, 0, -1):

        x = vec[i - 1]
        y = vec[i]

        norm = np.sqrt(abs(x) ** 2 + abs(y) ** 2)

        if np.isclose(norm, 0):
            continue

        A = align(x, y, norm)

        result.append(
            TwoLevel(
                size=N,
                level0=i - 1,
                level1=i,
                unitary=A,
            )
        )

        vec[i - 1] = norm
        vec[i] = 0

    return result


def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    """Expand each TwoLevel to n dimensions by shifting size, level0, level1 up by
    the offset (n - tl.size). Used to lift a sub-block decomposition back to full n.
    """
    # TODO: implement.
    output = []

    for tl in input:
        offset = n - tl.size

        output.append(
            TwoLevel(
                size=n,
                level0=tl.level0 + offset,
                level1=tl.level1 + offset,
                unitary=tl.unitary.copy()
            )
        )

    return output


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    """Full matrix of a two-level sequence: premultiply each two-level's matrix in
    order (result = tl.to_unitary() @ result), reproducing the application order.
    """
    # TODO: implement.
    if len(two_levels) == 0:
        raise ValueError("TwoLevel sequence must not be empty.")

    N = two_levels[0].size
    U = np.eye(N, dtype=rotation.DTYPE)

    for tl in two_levels:
        U = tl.to_unitary() @ U

    return U


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    """Adjoint of a single two-level: same levels, adjoint (conjugate transpose) of
    the 2x2 block.
    """
    # TODO: implement.
    return TwoLevel(
        size=tl.size,
        level0=tl.level0,
        level1=tl.level1,
        unitary=tl.unitary.conj().T
    )

def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    """Adjoint of a sequence: reverse the order and take the adjoint of each, since
    (A_k ... A_1)^dagger = A_1^dagger ... A_k^dagger.
    """
    # TODO: implement.
    result = []

    for tl in reversed(two_levels):
        result.append(adjoint_twolevel(tl))

    return result


def decompose_unitary(u: np.ndarray) -> TwoLevels:
    """Repeat decompose_vector on successive sub-columns to reduce u to identity.
    At step k, columns/rows 0..k-1 are already reduced, so work on the lower-right
    (n-k) block: clear column k below the diagonal. Finally append a phase two-level
    on the last two levels to cancel the residual phase, so the product is identity.
    Returns the sequence S with prod(S) @ u == I (i.e. prod(S) = u^dagger).
    """
    # TODO: implement.
    work = u.copy()

    N = work.shape[0]

    result = []

    for k in range(N-1):

        sub = work[k:, k:]

        tls = decompose_vector(sub[:,0])

        tls = expand_twolevels(tls, N)

        for tl in tls:
            work = tl.to_unitary() @ work
            result.append(tl)

    #
    # Remove remaining phase
    #

    phase = np.angle(work[-1,-1])

    if not np.isclose(phase,0):

        correction = np.array([
            [np.exp(-1j*phase),0],
            [0,1]
        ],dtype=rotation.DTYPE)

        result.append(
            TwoLevel(
                size=N,
                level0=N-2,
                level1=N-1,
                unitary=correction
            )
        )

    return result


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    """The two-level decomposition of u itself: decompose_unitary returns the
    sequence S that reduces u to identity (prod(S) = u^dagger), so its adjoint is
    the sequence whose product is u.
    """
    # TODO: implement (hint: adjoint_twolevels(decompose_unitary(u))).
    return adjoint_twolevels(decompose_unitary(u))


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate (see cpp/src/ABC.h)
# ---------------------------------------------------------------------------


@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)


def abc_decompose(u: np.ndarray) -> ABC:
    """Build the ABC decomposition of u (Corollary 4.2). Take the ZYZ Euler angles
    (alpha, beta, gamma, delta) of u, then set
        A = Rz(beta) Ry(gamma/2)
        B = Ry(-gamma/2) Rz(-(delta+beta)/2)
        C = Rz((delta-beta)/2)
    Using X Ry(t) X = Ry(-t) and X Rz(t) X = Rz(-t), these satisfy A B C = I and
    e^{i alpha} A X B X C = u.
    """
    # TODO: implement using rotation.euler_angles_zyz, rotation.Rz/Ry.
    alpha, beta, gamma, delta = rotation.euler_angles_zyz(u)

    A = rotation.Rz(beta) @ rotation.Ry(gamma / 2)

    B = (
        rotation.Ry(-gamma / 2)
        @ rotation.Rz(-(delta + beta) / 2)
    )

    C = rotation.Rz((delta - beta) / 2)

    return ABC(
        alpha=alpha,
        A=A,
        B=B,
        C=C
    )


def abc_reconstruct(d: ABC) -> np.ndarray:
    """Reassemble e^{i alpha} A X B X C from an ABC (inverse of abc_decompose)."""
    # TODO: implement.
    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=rotation.DTYPE)

    return (
        np.exp(1j * d.alpha)
        * d.A
        @ X
        @ d.B
        @ X
        @ d.C
    )


# ---------------------------------------------------------------------------
# Gray code and controlled circuits (see cpp/src/Swap.h, cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def gray_code(tl: TwoLevel) -> list[Swap]:
    """Gray code connecting level0 and level1 of a two-level, as the sequence of
    single-qubit flips walking from level0 to level1 (one Swap per differing bit).
    At each step the Swap records which qubit flips and the current code's values on
    the other qubits (the control pattern).
    """
    # TODO: implement.
    n = num_qubits(tl.size)

    current = tl.level0
    target = tl.level1

    swaps = []

    diff = current ^ target

    for bit in range(n):

        if diff & (1 << bit):

            controls = []

            for q in range(n):
                controls.append(bool((current >> q) & 1))

            swaps.append(
                Swap(
                    target=bit,
                    control_vals=controls
                )
            )

            current ^= (1 << bit)

    return swaps

def decompose_swap(swap: Swap) -> Circuit:
    """Decompose a Swap (multi-controlled NOT) into a Circuit: a controlled-X with
    the swap's arbitrary control values.
    """
    # TODO: implement (hint: controlled_circuit with Pauli-X).
    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=rotation.DTYPE)

    return controlled_circuit(
        n=len(swap.control_vals),
        target=swap.target,
        control_vals=swap.control_vals,
        unitary=X,
    )


def controlled_circuit(
    n: int, target: int, control_vals: list[bool], unitary: np.ndarray
) -> Circuit:
    """Circuit applying the 2x2 `unitary` to `target` iff every non-target qubit
    equals control_vals[q]. Realized as a fully-controlled-U core (ControlledU,
    controls = all 1) sandwiched by X gates on the qubits conditioned on 0, so those
    become 1-controls. The sandwich is symmetric (X is its own inverse).
    """
    # TODO: implement.
    circuit = []

    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=rotation.DTYPE)

    # Apply X gates to convert 0-controls into 1-controls
    for q in range(n):
        if q != target and not control_vals[q]:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X
                )
            )

    # Fully-controlled unitary
    circuit.append(
        ControlledU(
            n=n,
            target=target,
            unitary=unitary
        )
    )

    # Undo the temporary X gates
    for q in reversed(range(n)):
        if q != target and not control_vals[q]:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X
                )
            )

    return circuit


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline (see cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Lower a TwoLevel to a Circuit (Nielsen-Chuang 4.5.2): walk a gray code so
    level0 becomes adjacent to level1, apply the controlled-U on that last
    transition, then undo the walk. Orient the 2x2 so a00 (level0's corner) sits on
    the target value the second-to-last code has.
    """
    # TODO: implement using gray_code, decompose_swap, controlled_circuit.
    swaps = gray_code(tl)

    circuit = []

    # Forward walk
    for s in swaps[:-1]:
        circuit.extend(decompose_swap(s))

    
    last = swaps[-1]

    circuit.extend(
        controlled_circuit(
            n=num_qubits(tl.size),
            target=last.target,
            control_vals=last.control_vals,
            unitary=tl.unitary,
        )
    )
    

    # Undo walk
    for s in reversed(swaps[:-1]):
        circuit.extend(decompose_swap(s))

    return circuit


def decompose_controlled(
    n: int, controls: list[int], target: int, u: np.ndarray
) -> Circuit:
    """Decompose C^k(U) (k = len(controls)) into singly-controlled gates C(U) and
    CNOTs (Nielsen-Chuang fig 4.8). Base cases: no control -> a plain SingleQubitGate;
    one control -> a CNOT if U == X else a CU. Otherwise, with V = sqrt(U):
        a. C(V) on target
        b. C^{k-1}(X) onto the pivot control
        c. C(V dagger) on target
        d. repeat b
        e. C^{k-1}(V) on target
    Phases are kept throughout.
    """
    # TODO: implement (recursive; use rotation.unitary2_sqrt for V).
    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=rotation.DTYPE)

    # No controls
    if len(controls) == 0:
        return [
            SingleQubitGate(
                n=n,
                qubit=target,
                unitary=u
            )
        ]

    # One control
    if len(controls) == 1:
        c = controls[0]

        if np.allclose(u, X):
            return [
                CNOT(
                    n=n,
                    control=c,
                    target=target
                )
            ]

        return [
            CU(
                n=n,
                control=c,
                target=target,
                unitary=u
            )
        ]

    # Recursive case
    V = rotation.unitary2_sqrt(u)

    pivot = controls[-1]
    rest = controls[:-1]

    circuit = []

    circuit.extend(decompose_controlled(n, [pivot], target, V))
    circuit.extend(decompose_controlled(n, rest, pivot, X))
    circuit.extend(decompose_controlled(n, [pivot], target, V.conj().T))
    circuit.extend(decompose_controlled(n, rest, pivot, X))
    circuit.extend(decompose_controlled(n, rest, target, V))

    return circuit

def decompose_controlledU(g: ControlledU) -> Circuit:
    """Lower a ControlledU (controlled on all other qubits) into CNOTs + C(U): build
    the list of all non-target qubits as controls and call decompose_controlled.
    """
    # TODO: implement.
    controls = [
        q for q in range(g.n)
        if q != g.target
    ]

    return decompose_controlled(
        n=g.n,
        controls=controls,
        target=g.target,
        u=g.unitary,
    )


def decompose_cu(g: CU) -> Circuit:
    """Lower a singly-controlled C(U) into single-qubit gates + 2 CNOTs
    (Nielsen-Chuang Corollary 4.2 / fig 4.6). With U = e^{i alpha} A X B X C and
    A B C = I, emit: C, CNOT, B, CNOT, A on the target, plus a diag(1, e^{i alpha})
    phase on the control line. control=0: CNOTs vanish, target sees A B C = I;
    control=1: CNOTs act as X, target sees A X B X C = U with phase e^{i alpha}.
    """
    # TODO: implement using abc_decompose.
    abc = abc_decompose(g.unitary)

    phase = np.array([
        [1, 0],
        [0, np.exp(1j * abc.alpha)]
    ], dtype=rotation.DTYPE)

    return [
        SingleQubitGate(
            n=g.n,
            qubit=g.control,
            unitary=phase,
        ),

        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.C,
        ),

        CNOT(
            n=g.n,
            control=g.control,
            target=g.target,
        ),

        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.B,
        ),

        CNOT(
            n=g.n,
            control=g.control,
            target=g.target,
        ),

        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.A,
        ),
    ]


def decompose_to_basis(u: np.ndarray) -> Circuit:
    """Fully lower a Unitary to a Circuit of only SingleQubitGate and CNOT, running
    the four stages in sequence:
        1. twolevel_decomposition: Unitary     -> TwoLevels
        2. decompose_twolevel:     TwoLevel    -> SingleQubitGate + ControlledU
        3. decompose_controlledU:  ControlledU -> CU + CNOT
        4. decompose_cu:           CU          -> SingleQubitGate + CNOT
    Each stage rewrites only its own gate type and passes the rest through unchanged.
    """
    # TODO: implement (run each rewrite pass over the circuit).
    # Stage 1
    circuit = to_circuit(twolevel_decomposition(u))

    # Stage 2
    new_circuit = []
    for gate in circuit:
        if isinstance(gate, TwoLevel):
            new_circuit.extend(decompose_twolevel(gate))
        else:
            new_circuit.append(gate)
    circuit = new_circuit

    # Stage 3
    new_circuit = []
    for gate in circuit:
        if isinstance(gate, ControlledU):
            new_circuit.extend(decompose_controlledU(gate))
        else:
            new_circuit.append(gate)
    circuit = new_circuit

    # Stage 4
    new_circuit = []
    for gate in circuit:
        if isinstance(gate, CU):
            new_circuit.extend(decompose_cu(gate))
        else:
            new_circuit.append(gate)
    circuit = new_circuit

    return circuit


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    """Expand a flat H/T word into a Circuit of SingleQubitGate H/T gates on `qubit`.
    The word (leftmost char = leftmost matrix factor) is pushed in reverse so the
    circuit's application order (first gate first = rightmost factor) reproduces
    rotation.gates_to_unitary(word).
    """
    # TODO: implement.
    circuit = []

    T = np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ], dtype=rotation.DTYPE)

    for gate in reversed(word):

        if gate == "H":
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=qubit,
                    unitary=rotation.H,
                )
            )

        elif gate == "T":
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=qubit,
                    unitary=T,
                )
            )

        else:
            raise ValueError(f"Unknown gate '{gate}'")

    return circuit


def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    """Fully lower a Unitary to a Circuit of only H, T, and CNOT gates (the discrete
    fault-tolerant basis): run decompose_to_basis, then replace each arbitrary
    SingleQubitGate with its {H, T} word from rotation.approximate_in_ht (CNOTs pass
    through). `error` is the per-gate angular tolerance (smaller -> longer, more
    accurate). Each word matches its gate up to a global phase; those per-gate phases
    factor out into one overall global phase, so the result reconstructs u up to
    global phase (compare with error_up_to_phase).
    """
    # TODO: implement using decompose_to_basis, ht_gates, and rotation.approximate_in_ht.
    basis = decompose_to_basis(u)

    circuit = []

    for gate in basis:

        if isinstance(gate, SingleQubitGate):

            word = rotation.approximate_in_ht(
                gate.unitary,
                error,
            )

            circuit.extend(
                ht_gates(
                    gate.n,
                    gate.qubit,
                    word,
                )
            )

        else:
            # CNOT passes through unchanged
            circuit.append(gate)

    return circuit
