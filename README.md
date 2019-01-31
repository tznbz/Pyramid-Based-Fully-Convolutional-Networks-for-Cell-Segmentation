# Pyramid-Based-Fully-Convolutional-Networks-for-Cell-Segmentation
The low contrast and irregular cell shapes in microscopy images cause difficulties to obtain the accurate cell segmentation. We propose pyramid-based fully convolutional networks (FCN) to segment cells in a cascaded refinement manner. The higher-level FCNs generate coarse cell segmentation masks, attacking the challenge of low contrast between cell inner regions and the background. The lower-level FCNs generate segmentation masks focusing more on cell details, attacking the challenge of irregular cell shapes. The FCNs in the pyramid are trained in a cascaded way such that the residual error between the ground truth and upper-level segmentation is propagated to the lower-level and draws the attention of the lower-level FCNs to find the cell details missed from the upper-levels. The fine cell details from lower-level FCNs are gradually fused into the coarse segmentation from upper-level FCNs so as to obtain a final precise cell segmentation mask.


## Reference
    @paper{zhao2018pyramid,
    title={Pyramid-Based Fully Convolutional Networks for Cell Segmentation},
    author={Zhao, Tianyi and Yin, Zhaozheng},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, 
    pages={677--685},
    year={2018},
    organization={Springer}
    }
 [[SpringerLink]](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_77)



## Dataset
 (http://celltrackingchallenge.net/2d-datasets/)
