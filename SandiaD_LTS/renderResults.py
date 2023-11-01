# trace generated using paraview version 5.10.0
import paraview, argparse, os
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

parser = argparse.ArgumentParser(description='Script to render an image of the SandiaD_LTS case')
parser.add_argument('--decomposed', action='store_true', help='case is decomposed')
parser.add_argument('image', help='image name')
args = parser.parse_args()

# create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=f'{os.getcwd()}/case.foam')
casefoam.MeshRegions = ['internalMesh']
casefoam.Decomposepolyhedra = 0

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
casefoamDisplay = Show(casefoam, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
#casefoamDisplay.Representation = 'Surface'
#casefoamDisplay.ColorArrayName = [None, '']
#casefoamDisplay.SelectTCoordArray = 'None'
#casefoamDisplay.SelectNormalArray = 'None'
#casefoamDisplay.SelectTangentArray = 'None'
#casefoamDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
#casefoamDisplay.SelectOrientationVectors = 'None'
#casefoamDisplay.ScaleFactor = 0.06000000014901161
#casefoamDisplay.SelectScaleArray = 'None'
#casefoamDisplay.GlyphType = 'Arrow'
#casefoamDisplay.GlyphTableIndexArray = 'None'
#casefoamDisplay.GaussianRadius = 0.0030000000074505806
#casefoamDisplay.SetScaleArray = [None, '']
#casefoamDisplay.ScaleTransferFunction = 'PiecewiseFunction'
#casefoamDisplay.OpacityArray = [None, '']
#casefoamDisplay.OpacityTransferFunction = 'PiecewiseFunction'
#casefoamDisplay.DataAxesGrid = 'GridAxesRepresentation'
#casefoamDisplay.PolarAxes = 'PolarAxesRepresentation'
#casefoamDisplay.ScalarOpacityUnitDistance = 0.05013964280123844
#casefoamDisplay.OpacityArrayName = ['FIELD', 'CasePath']

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# reset view to fit data
renderView1.ResetCamera(False)

#change interaction mode for render view
renderView1.InteractionMode = '2D'

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# Properties modified on casefoam
if args.decomposed:
    casefoam.CaseType = 'Decomposed Case'

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on casefoam
casefoam.SkipZeroTime = 0
casefoam.CellArrays = ['C2H3', 'C2H4', 'C2H5', 'C2H6', 'C3H7', 'C3H8', 'CH2', 'CH2(S)', 'CH2CHO', 'CH2CO', 'CH2O', 'CH2OH', 'CH3', 'CH3CHO', 'CH3O', 'CH3OH', 'CH4', 'CO', 'CO2', 'EDC<psiReactionThermo>:kappa', 'G', 'H', 'H2', 'H2O', 'H2O2', 'HCO', 'HO2', 'N2', 'O', 'O2', 'OH', 'Qdot', 'T', 'TabulationResults', 'U', 'a', 'alphat', 'epsilon', 'k', 'nut', 'p', 'qr', 'rDeltaT']

# update the view to ensure updated data information
renderView1.Update()

animationScene1.GoToLast()

# set scalar coloring
ColorBy(casefoamDisplay, ('CELLS', 'CO2'))

# rescale color and/or opacity maps used to include current data range
casefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
casefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'CO2'
cO2LUT = GetColorTransferFunction('CO2')

# get opacity transfer function/opacity map for 'CO2'
cO2PWF = GetOpacityTransferFunction('CO2')

animationScene1.GoToLast()

# get color legend/bar for cO2LUT in view renderView1
cO2LUTColorBar = GetScalarBar(cO2LUT, renderView1)

# change scalar bar placement
cO2LUTColorBar.WindowLocation = 'Any Location'
cO2LUTColorBar.Position = [0.8477987421383648, 0.3132530120481928]
cO2LUTColorBar.ScalarBarLength = 0.3300000000000002

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=casefoam)

# Properties modified on casefoam
casefoam.CellArrays = ['C2H3', 'C2H4', 'C2H5', 'C2H6', 'C3H7', 'C3H8', 'CH2', 'CH2(S)', 'CH2CHO', 'CH2CO', 'CH2O', 'CH2OH', 'CH3', 'CH3CHO', 'CH3O', 'CH3OH', 'CH4', 'CO', 'CO2', 'EDC<psiReactionThermo>:kappa', 'G', 'H', 'H2', 'H2O', 'H2O2', 'HCO', 'HO2', 'N2', 'O', 'O2', 'OH', 'Qdot', 'T', 'TabulationResults', 'U', 'a', 'alphat', 'epsilon', 'k', 'nut', 'p', 'qr', 'rDeltaT', 'Ydefault', 'C2H', 'C2H2', 'CH', 'HCCO', 'HCCOH']

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1, 'TextSourceRepresentation')
annotateTimeFilter1Display.Position = [0.84, 0.71]

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Format = 'Time: {time:0f}s'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.WindowLocation = 'Any Location'

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Format = 'Time: {time:.0f}s'

# update the view to ensure updated data information
renderView1.Update()

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1590, 830)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.07500000298023224, -1.1950521900117717, 0.19999999925494194]
renderView1.CameraFocalPoint = [0.07500000298023224, 0.0, 0.19999999925494194]
renderView1.CameraViewUp = [1.0, 0.0, 2.220446049250313e-16]
renderView1.CameraParallelScale = 0.1574447667125202

# save screenshot
SaveScreenshot(f'{os.getcwd()}/{args.image}.png', renderView1, ImageResolution=[1590, 830],
    TransparentBackground=1, 
    # PNG options
    CompressionLevel='3')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1590, 830)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.07500000298023224, -1.1950521900117717, 0.19999999925494194]
renderView1.CameraFocalPoint = [0.07500000298023224, 0.0, 0.19999999925494194]
renderView1.CameraViewUp = [1.0, 0.0, 2.220446049250313e-16]
renderView1.CameraParallelScale = 0.1574447667125202

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
