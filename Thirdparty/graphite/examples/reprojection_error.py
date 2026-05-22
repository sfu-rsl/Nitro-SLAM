from wrenfold import code_generation, sym, geometry
from wrenfold.type_annotations import Vector2, Vector3, FloatScalar
from wrenfold.geometry import Quaternion
def projection_simple(rvec: Vector3, t: Vector3, f: FloatScalar, k1: FloatScalar, k2: FloatScalar, point: Vector3):
    """
    Project a 3D point onto a 2D image plane using a pinhole camera model with radial distortion.
    
    Args:
        R: Rotation vector (3D).
        t: Translation vector (3D).
        f: Focal length.
        k1: Radial distortion coefficient 1.
        k2: Radial distortion coefficient 2.
    
    Returns:
        The projected 2D point on the image plane.
    """


    theta = rvec.norm()

    Q = Quaternion.from_rotation_vector(rvec[0], rvec[1], rvec[2], None) 
    R_matrix = sym.where(
        sym.gt(theta, 0.0),
        Q.to_rotation_matrix(),
        sym.eye(3)
    )

    P = R_matrix*point + t
    
    P = -Vector2([P[0], P[1]]) / P[2]

    r2 = P[0]**2 + P[1]**2

    radial_distortion = 1.0 + k1 * r2 + k2 * r2*r2

    proj = f*radial_distortion*P

    Jc = sym.jacobian([proj[0], proj[1]], [rvec[0], rvec[1], rvec[2], t[0], t[1], t[2], f, k1, k2])
    Jp = sym.jacobian([proj[0], proj[1]], [point[0], point[1], point[2]])
    
    return (
        # code_generation.ReturnValue(proj),
        code_generation.OutputArg(Jc, "Jc"),
        code_generation.OutputArg(Jp, "Jp")
    )



cpp = code_generation.generate_function(projection_simple, code_generation.CppGenerator(code_generation.CppMatrixTypeBehavior.Eigen))
# print(cpp)
with open("projection_jacobians.cuh", "w") as f:
    f.write(cpp)

print("OK")