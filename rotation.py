import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""

    # 1. Extract the global phase using the determinant
    # det(G) = e^{i * 2 * alpha}
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    alpha = np.angle(det) / 2.0

    # 2. Factor out the global phase to isolate the SU(2) rotation part
    u = g * np.exp(-1j * alpha)

    # 3. Extract the rotation angle theta
    # u[0, 0] = cos(theta/2) - i * nz * sin(theta/2)
    cos_half = np.real(np.trace(u)) / 2.0
    
    # Clip to [-1, 1] to avoid nan in arccos from tiny floating point inaccuracies
    cos_half = np.clip(cos_half, -1.0, 1.0) 
    theta = 2.0 * np.arccos(cos_half)

    # 4. Extract the rotation axis n = [nx, ny, nz]
    sin_half = np.sin(theta / 2.0)
    
    if np.isclose(sin_half, 0.0):
        # Rotation is a multiple of 2*pi, axis is arbitrary
        n = np.array([1.0, 0.0, 0.0])
    else:
        # u[0, 1] = -ny * sin(theta/2) - i * nx * sin(theta/2)
        nx = -np.imag(u[0, 1]) / sin_half
        ny = -np.real(u[0, 1]) / sin_half
        nz = -np.imag(u[0, 0]) / sin_half
        
        n = np.array([nx, ny, nz])
        
        # Normalize to ensure it's a perfect unit vector
        norm = np.linalg.norm(n)
        if norm > 0:
            n = n / norm
        else:
            n = np.array([1.0, 0.0, 0.0])

    # 5. Populate and return the Bloch object
    b = Bloch()
    b.alpha = float(alpha)
    b.n = n
    b.theta = float(theta)
    return b
# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# Standard X and Y axes are chosen here for the standard Euler decomposition.
c = np.cos(np.pi / 8)

n1 = np.array([
    -c,
     1.0,
     c
], dtype=float)
n1 /= np.linalg.norm(n1)

n2 = np.array([
     1.0 / np.sqrt(2.0),
     np.sqrt(2.0) * c,
    -1.0 / np.sqrt(2.0)
], dtype=float)
n2 /= np.linalg.norm(n2)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)
a3 /= np.linalg.norm(a3)

def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    c = np.cos(b.theta / 2.0)
    s = np.sin(b.theta / 2.0)

    x = s * np.dot(b.n, a1)
    y = s * np.dot(b.n, a2)
    z = s * np.dot(b.n, a3)

    beta = 2.0 * np.arccos(
        np.clip(np.sqrt(c*c + x*x), 0.0, 1.0)
    )

    sum_angle = np.arctan2(x, c)
    diff_angle = np.arctan2(z, y)

    alpha = (sum_angle - diff_angle) / 2.0
    gamma = (sum_angle + diff_angle) / 2.0

    alpha %= TWO_PI
    beta %= TWO_PI
    gamma %= TWO_PI

    return float(alpha), float(beta), float(gamma), float(b.alpha)


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    angle %= TWO_PI

    MAX_K = 10_000_000

    for k in range(1,MAX_K):
        current = (k * LAMBDA_PI) % TWO_PI

        diff = abs(current - angle)
        diff = min(diff, TWO_PI - diff)

        if diff <= tolerance:
            return k

    raise RuntimeError("No approximation found within search limit.")


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
     # Step 1: Convert to Bloch form
    bloch = to_bloch(u)

    # Step 2: Find the Euler angles
    alpha, beta, gamma, _ = n1n2n1_angles(bloch)

    # Step 3: Approximate each angle by an integer multiple of λπ
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    # Step 4: Return the powers
    return (k, l, m)
