from   scipy.io   import  loadmat
import numpy      as      np
matrixVar = loadmat( "data/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat" )

# Do whatever data manipulation you need here
# Let's do a simple transpose for the sake of the example.
mainpulatedData = np.transpose(matrixVar)

# Do more stuff here if needed
# ...
print( 'Done processing' )