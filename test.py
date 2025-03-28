import numpy as np
import mathutils

def interpolate_affine_numpy7(A, B, steps=10):
    """
    Interpolates between two affine transformations represented as 7-element NumPy arrays.
    
    Each input array should be of shape (7,) where:
      - A[0:3] represents the translation (x, y, z)
      - A[3:7] represents the rotation as a quaternion (w, x, y, z)
    
    The function returns a list of (steps+1) NumPy arrays that include the start and end transforms.
    For example, if steps=10, you get 11 transformations (A, 9 intermediates, B).
    
    Args:
        A (np.ndarray): A 7-element array for the starting transformation.
        B (np.ndarray): A 7-element array for the ending transformation.
        steps (int): Number of intermediate steps (default 10).
    
    Returns:
        List[np.ndarray]: A list of 7-element NumPy arrays representing interpolated transforms.
    """
    if A.shape != (7,) or B.shape != (7,):
        raise ValueError("Both A and B must be numpy arrays of shape (7,)")
    
    # Extract translation components
    transA = np.array(A[:3])
    transB = np.array(B[:3])
    
    # Extract quaternion components and create mathutils.Quaternion instances.
    quatA = mathutils.Quaternion(A[3:7])
    quatB = mathutils.Quaternion(B[3:7])
    
    result = []
    # Create (steps + 1) interpolated transformations, including endpoints.
    for i in range(steps + 1):
        t = i / float(steps)
        
        # Interpolate translation linearly.
        trans_interp = (1 - t) * transA + t * transB
        
        # Interpolate rotation using slerp.
        # We copy quatA to avoid modifying it and then slerp towards quatB.
        quat_interp = mathutils.Quaternion.slerp(quatA, quatB, t)

        # Build the 7-element array: translation followed by quaternion in (w,x,y,z) order.
        interp_array = np.concatenate([trans_interp, np.array([quat_interp.w, quat_interp.x, quat_interp.y, quat_interp.z])])
        result.append(interp_array)
    
    return result

# Example usage:
if __name__ == "__main__":
    # Define two example affine transformations (7-element arrays).
    # A: translation (1,2,3) and quaternion (1,0,0,0) (identity rotation)
    # B: translation (4,5,6) and quaternion (0.7071, 0, 0.7071, 0) (rotation of 90 degrees around Y axis)
    A = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    B = np.array([4.0, 5.0, 6.0, 0.7071, 0.0, 0.7071, 0.0])
    
    transforms = interpolate_affine_numpy7(A, B, steps=10)
    
    # Print out the interpolated transformations.
    for i, tform in enumerate(transforms):
        print(f"Step {i}: {tform}")
