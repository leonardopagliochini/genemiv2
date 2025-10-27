import SVMTK as svmtk
import time

# Import surfaces, and merge lh/rh white surfaces
breast  = svmtk.Surface("segmentations/Segmentation_Breast_MRI_076_Breast_Breast.stl") 
dense = svmtk.Surface("segmentations/Segmentation_Breast_MRI_076_Dense_and_Vessels_Dense.stl") 
vessels = svmtk.Surface("segmentations/Segmentation_Breast_MRI_076_Dense_and_Vessels_Vessels.stl")

surfaces = [breast, dense, vessels] 

# Create subdomain map
smap = svmtk.SubdomainMap() 
smap.add("100", 1)
smap.add("110", 2) 
smap.add("101", 3)
smap.add("111", 3)

# Create domain
domain = svmtk.Domain(surfaces, smap)

# Create meshes of increasing resolutions
Ns = [256]
for N in Ns: 
    print("Creating mesh for N=%d" % N)
    t0 = time.time()
    domain.create_mesh(N) 
    domain.remove_subdomain([3]) 
    domain.save("breast_%d.mesh" % N)
    t1 = time.time()
    print("Done! That took %g sec" % (t1-t0))
