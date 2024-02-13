from keypoint_transformer.call_kypt_transformer import kypt_trafo_for_covariance, kypt_trafo_for_covariance_test
import numpy as np
from surface_mesh_methods.object_functions import rot_param_rot_mat
import torch
import pytransform3d.rotations as pr
import pytransform3d.transformations._conversions as ptc
import pytransform3d.uncertainty as pu        
from scipy.linalg import polar
from pytransform3d.transformations import (
    invert_transform, transform_from_exponential_coordinates)
from pytransform3d.trajectories import (exponential_coordinates_from_transforms,
                           concat_many_to_one)



def main():
    train = 1
    if train:
        (kypt_obj_rot, kypt_obj_trans, obj_rotation_annotated_vec, 
        obj_translation_annotated_vec) = kypt_trafo_for_covariance()
        covariance_vec = []
        trans_covariance_vec = []

        for i in range(len(kypt_obj_trans)):
            kypt_obj_rot_tensor = torch.from_numpy(kypt_obj_rot[i])
            rotation_matrix_estimated = rot_param_rot_mat(kypt_obj_rot_tensor.reshape(-1, 6))[0].cpu().numpy()

            diff_trans = kypt_obj_trans[i] - obj_translation_annotated_vec[i]
            trans_covariance_vec.append(diff_trans)

            transformation_matrix_estimated = np.eye(4)
            transformation_matrix_estimated[:3, :3] = rotation_matrix_estimated
            
            improve_rotation_matrix = 1

            if improve_rotation_matrix:
                rotation_matrix_estimated = sharp_rotation_matrix(rotation_matrix_estimated)
            
            transformation_matrix_estimated[:3, -1] = kypt_obj_trans[i]
            transformation_matrix_estimated = create_transformation_matrix(rotation_matrix_estimated, kypt_obj_trans[i])

            rotation_matrix_annotated = pr.matrix_from_euler(obj_rotation_annotated_vec[i], 0, 1, 2, True)
            transformation_matrix_annotated = create_transformation_matrix(rotation_matrix_annotated, obj_translation_annotated_vec[i])

            transformation_matrix_difference = np.dot(np.linalg.inv(transformation_matrix_annotated), transformation_matrix_estimated).reshape((4, 4))

            covariance_vec.append(transformation_matrix_difference)

        covariance_mat = estimate_gaussian_transform_from_samples(covariance_vec)
        print("covariance_mat:\n", covariance_mat)
    
    else:
        kypt_obj_trans, obj_translation_annotated_vec = kypt_trafo_for_covariance_test()
        covariance_vec = []
        trans_covariance_vec = []
        for i in range(len(kypt_obj_trans)):
            diff_trans = kypt_obj_trans[i] - obj_translation_annotated_vec[i]
            trans_covariance_vec.append(diff_trans)

    cov_matrix = np.cov(trans_covariance_vec, rowvar=False, bias=True)
    print("Covariance Matrix (3x3):", cov_matrix)

    standard_deviation = np.sqrt(np.diag(cov_matrix))
    print("Standard deviation for each variable:", standard_deviation)


def sharp_rotation_matrix(matrix, max_iterations=100, tolerance=1e-6):
    # initialize the iteration counter and the error
    iteration = 0
    error = np.inf
    print("matrix:\n", matrix)
    # iterate until convergence or until the maximum number of iterations is reached
    while error > tolerance and iteration < max_iterations:
        # compute the polar decomposition of the matrix
        U, P = polar(matrix)
        # compute the error between the original matrix and the improved matrix
        error = np.linalg.norm(matrix - np.dot(U, P))
        # update the matrix with the improved version
        matrix = np.dot(U, P)
        iteration += 1
    print("Number of iterations:", iteration)
    print("Error:", error)
    print("Improved matrix:\n", matrix)

    return matrix


def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, -1] = translation_vector
    return transformation_matrix   


def estimate_gaussian_transform_from_samples(samples):
    """Estimate Gaussian distribution over transformations from samples.

    Uses iterative approximation of mean described by Eade (2017) and computes
    covariance in exponential coordinate space (using an unbiased estimator).

    Parameters
    ----------
    samples : array-like, shape (n_samples, 4, 4)
        Sampled transformations represented by homogeneous matrices.

    Returns
    -------
    mean : array, shape (4, 4)
        Mean as homogeneous transformation matrix.

    cov : array, shape (6, 6)
        Covariance of distribution in exponential coordinate space.

    References
    ----------
    Eade: Lie Groups for 2D and 3D Transformations (2017),
    https://ethaneade.com/lie.pdf
    """
    assert len(samples) > 0
    mean = samples[0]
    for _ in range(20):
        mean_inv = invert_transform(mean, strict_check=False, check=False)  # changed
        transformations_with_inverse = concat_many_to_one(samples, mean_inv)
        mean_diffs = exponential_coordinates_from_transforms(transformations_with_inverse) 
        avg_mean_diff = np.mean(mean_diffs, axis=0)
        mean = np.dot(
            transform_from_exponential_coordinates(avg_mean_diff), mean) 

    cov = np.cov(mean_diffs, rowvar=False, bias=True)
    return mean, cov


if __name__ == "__main__":
    main()

