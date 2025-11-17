import SVMTK as svmtk
import time

# Import surfaces, and merge lh/rh white surfaces
breast  = svmtk.Surface("segmentations/new_skin.stl") 
dense = svmtk.Surface("segmentations/new_fat.stl") 

surfaces = [breast, dense] 

# Create subdomain map
smap = svmtk.SubdomainMap() 
smap.add("10", 1)
smap.add("11", 2)

# Create domain
domain = svmtk.Domain(surfaces, smap)

# Create meshes of increasing resolutions
Ns = [128]
for N in Ns: 
    print("Creating mesh for N=%d" % N)
    t0 = time.time()
    domain.create_mesh(N) 
    # domain.remove_subdomain([3]) 
    domain.save("breast_%d.mesh" % N)
    t1 = time.time()
    print("Done! That took %g sec" % (t1-t0))
