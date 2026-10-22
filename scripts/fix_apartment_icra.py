"""Fixes for IAI's apartment USD (apartmentICRA.usda, exported from Unreal
through Omniverse), written as a layer over it (the original is left as it
is): apartmentICRA_fixed.usda next to it.

  - Unreal's default WorldGridMaterial (a missing material) is on part of
    three living-room walls and two sockets: bound to the material the other
    walls (M_Plastic_White) and sockets (M_Metal_Chrome) have there;
  - the wall cabinet's two flaps (SM_Kitchen_04_Door_01 / _02) and the oven
    door are static: rigid bodies on revolute joints like the other doors,
    the flaps hinged at their top edge, the oven door at its bottom, each
    opening 90 degrees outwards;
  - the standing table's foot (SM_Stehtisch) stands 6 cm into the dining
    table's leg: moved 8 cm away from it.

    python3 scripts/fix_apartment_icra.py tmp/apartment/apartmentICRA.usda
    python3 scripts/usd2mjcf.py tmp/apartment/apartmentICRA_fixed.usda \\
        mjcf_assets/environments/aicor/iai_apartment_usd --offset 0 0 0.07 \\
        --no-collision 'Shadow[bB]ox'

(SM_Living_Shadowbox is the building's shell: ceiling, roof, outer walls;
SM_Window_ShadowBox the facade outside the windows. Without collision
geoms of their own, their bounding boxes would hold the whole apartment.)
"""

import pathlib
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = '/apartmentICRA'
# (mesh with WorldGridMaterial, the prim whose material of the same section to use)
MATERIALS = {
  'SM_Living_Wall_02': ('SM_Wall_03', 'M_Plastic_White'),
  'SM_Living_Wall_03': ('SM_Wall_03', 'M_Plastic_White'),
  'SM_Living_Wall_04': ('SM_Wall_03', 'M_Plastic_White'),
  'SM_Socket_030': ('SM_Socket_011', 'M_Metal_Chrome'),
  'SM_Socket_031': ('SM_Socket_011', 'M_Metal_Chrome'),
}
# door: (frame, hinge edge, axis sign, mass): the hinge runs along y at the
# door's front (min x) and its top or bottom.
DOORS = {
  'SM_Kitchen_04_Door_01': ('SM_Kitchen_04_Base', 'top', 1, 2.0),
  'SM_Kitchen_04_Door_02': ('SM_Kitchen_04_Base', 'top', 1, 2.0),
  'SM_Oven_Door': ('SM_Oven', 'bottom', -1, 4.0),
}
MOVES = {'SM_Stehtisch': (0.08, 0, 0)}   # m, in the world


def main():
  src = pathlib.Path(sys.argv[1]).resolve()
  out = src.with_name(src.stem + '_fixed.usda')
  if out.exists():
    out.unlink()
  layer = Sdf.Layer.CreateNew(str(out))
  layer.subLayerPaths.append(f'./{src.name}')
  fixed = Usd.Stage.Open(layer)
  fixed.SetEditTarget(layer)
  fixed.SetDefaultPrim(fixed.GetPrimAtPath(ROOT))
  UsdGeom.SetStageUpAxis(fixed, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(fixed, 1.0)

  for name, (other, material) in MATERIALS.items():
    mesh = fixed.GetPrimAtPath(f'{ROOT}/{name}/{name}')
    target = UsdShade.Material(fixed.GetPrimAtPath(f'{ROOT}/{other}/{other}/Materials/{material}'))
    for subset in mesh.GetChildren():
      bound, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
      if bound and bound.GetPrim().GetName() == 'WorldGridMaterial':
        UsdShade.MaterialBindingAPI.Apply(fixed.OverridePrim(subset.GetPath())).Bind(target)
        print(f'{subset.GetPath()}: {material}')

  cache = UsdGeom.XformCache()
  for name, (frame, edge, sign, mass) in DOORS.items():
    door = fixed.GetPrimAtPath(f'{ROOT}/{name}')
    base = fixed.GetPrimAtPath(f'{ROOT}/{frame}')
    panel = fixed.GetPrimAtPath(f'{door.GetPath()}/{name}')   # the door's own mesh, without its handle
    box = UsdGeom.Imageable(panel).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    hinge = Gf.Vec3d(lo[0], (lo[1] + hi[1]) / 2, hi[2] if edge == 'top' else lo[2])
    over = fixed.OverridePrim(door.GetPath())
    UsdPhysics.RigidBodyAPI.Apply(over)
    UsdPhysics.MassAPI.Apply(over).CreateMassAttr(mass)
    joint = UsdPhysics.RevoluteJoint.Define(fixed, f'{base.GetPath()}/{name}_joint')
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([door.GetPath()])
    joint.CreateAxisAttr(UsdPhysics.Tokens.y)
    # Opening outwards (-x): about +y for a top hinge, -y for a bottom one.
    joint.CreateLowerLimitAttr(0.0 if sign > 0 else -90.0)
    joint.CreateUpperLimitAttr(90.0 if sign > 0 else 0.0)
    for k, body in ((0, base), (1, door)):
      m = cache.GetLocalToWorldTransform(body)
      local = m.GetInverse().Transform(hinge)
      rot = Gf.Matrix4d(m).RemoveScaleShear().ExtractRotationQuat().GetInverse()
      getattr(joint, f'CreateLocalPos{k}Attr')(Gf.Vec3f(local))
      getattr(joint, f'CreateLocalRot{k}Attr')(Gf.Quatf(rot))
    print(f'{joint.GetPath()}: hinge at {tuple(round(v, 3) for v in hinge)}, {edge}')

  for name, d in MOVES.items():
    prim = fixed.GetPrimAtPath(f'{ROOT}/{name}')
    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]   # its xformOp:transform
    m = Gf.Matrix4d(op.Get())
    m.SetTranslateOnly(m.ExtractTranslation() + Gf.Vec3d(*d))
    op.Set(m)
    print(f'{prim.GetPath()}: moved by {d}')

  layer.Save()
  print(out)


if __name__ == '__main__':
  main()
