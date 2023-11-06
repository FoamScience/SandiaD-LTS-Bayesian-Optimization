# trace generated using paraview version 5.10.0
import paraview, argparse, os
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

from paraview.simple import *
import paraview.servermanager as servermanager
import pandas as pd
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

parser = argparse.ArgumentParser(description='Script to postProcess flame D data and compare to experiment')
parser.add_argument('--CO2', action='store_true', help='compare CO2 data')
parser.add_argument('--CH4', action='store_true', help='compare CH4 data')
parser.add_argument('--T', action='store_true', help='compare T data')
parser.add_argument('--U', action='store_true', help='compare U data')
parser.add_argument('--decomposed', action='store_true', help='case is decomposed')
args = parser.parse_args()

if  os.path.isfile(f"{os.getcwd()}/FAILED"):
    """If the case failed, penalize the objective and exit"""
    print("500000")
    exit()

def compare(x, y, xExp, yExp):
    yInterp = np.interp(xExp, x, y)
    yInterpNorm = yInterp/yInterp.max()
    yExpNorm = yExp/yExp.max()
    squaredErr = np.square(yExpNorm - yInterpNorm)
    return squaredErr.mean()

# create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=f'{os.getcwd()}/case.foam')
casefoam.MeshRegions = ['internalMesh']
casefoam.CellArrays = ['C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'C3H7', 'C3H8', 'CH', 'CH2', 'CH2(S)', 'CH2CHO', 'CH2CO', 'CH2O', 'CH2OH', 'CH3', 'CH3CHO', 'CH3O', 'CH3OH', 'CH4', 'CO', 'CO2', 'EDC<psiReactionThermo>:kappa', 'G', 'H', 'H2', 'H2O', 'H2O2', 'HCCO', 'HCCOH', 'HCO', 'HO2', 'N2', 'O', 'O2', 'OH', 'Qdot', 'T', 'TabulationResults', 'U', 'a', 'alphat', 'epsilon', 'k', 'nut', 'p', 'qr', 'rDeltaT']
casefoam.Decomposepolyhedra = 0
if args.decomposed:
    casefoam.CaseType = 'Decomposed Case'
UpdatePipeline(time=5000.0, proxy=casefoam)
animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=casefoam)
plotOverLine1.Point1 = [0.0, 0.0, -0.1]
plotOverLine1.Point2 = [0.0, 0.0, 0.5]
UpdatePipeline(time=5000.0, proxy=plotOverLine1)

# fetch data from pipeline
D = 7.2e-3
data = servermanager.Fetch(plotOverLine1)
df = pd.DataFrame(columns=['x/D', 'CO2'])
df['x/D'] = vtk_to_numpy(data.GetPointData().GetArray('arc_length'))/D
df['CO2'] = vtk_to_numpy(data.GetPointData().GetArray('CO2'))
df['CH4'] = vtk_to_numpy(data.GetPointData().GetArray('CH4'))
df['T'] = vtk_to_numpy(data.GetPointData().GetArray('T'))
U = vtk_to_numpy(data.GetPointData().GetArray('U'))
df['U'] = np.sqrt(U[:,0]**2 + U[:,1]**2 + U[:,2]**2)

if args.CO2:
    expDf = pd.read_csv(f"{os.getcwd()}/data/co2.csv")
    print(f"{compare(df['x/D'], df['CO2'], expDf['x/D'], expDf['CO2']):.3e}")
if args.CH4:
    expDf = pd.read_csv(f"{os.getcwd()}/data/ch4.csv")
    print(f"{compare(df['x/D'], df['CH4'], expDf['x/D'], expDf['CH4']):.3e}")
if args.T:
    expDf = pd.read_csv(f"{os.getcwd()}/data/T.csv")
    print(f"{compare(df['x/D'], df['T'], expDf['x/D'], expDf['T']):.3e}")
if args.U:
    expDf = pd.read_csv(f"{os.getcwd()}/data/U.csv")
    print(f"{compare(df['x/D'], df['U'], expDf['x/D'], expDf['U']):.3e}")
