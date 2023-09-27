from keypoint_transformer.call_kypt_transformer import kypt_trafo_for_covariance
import numpy as np


def main():
    (joints_right, joints_right_3d, joints_right_annotated) = kypt_trafo_for_covariance()

    covariance_mat = np.cov(joints_right_3d, joints_right_annotated[0])

    print("covariance between hand joints:\n", covariance_mat)


if __name__ == "__main__":
    main()