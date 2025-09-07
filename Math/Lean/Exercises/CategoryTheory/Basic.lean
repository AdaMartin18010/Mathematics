-- 范畴论基础练习 | Category Theory Basic Exercises
-- 对齐国际标准：剑桥大学Part III范畴论课程
-- 更新时间：2025-01-15

import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Functor.Basic
import Mathlib.CategoryTheory.NaturalTransformation

namespace CategoryTheoryExercises

-- 练习1：范畴的基本性质
-- 对应剑桥大学Part III范畴论标准
theorem category_comp_assoc (C : Type) [Category C] (X Y Z W : C)
  (f : X ⟶ Y) (g : Y ⟶ Z) (h : Z ⟶ W) :
  (f ≫ g) ≫ h = f ≫ (g ≫ h) := by
  exact Category.assoc f g h

-- 练习2：函子的基本性质
-- 对应哈佛大学范畴论标准
theorem functor_comp (C D : Type) [Category C] [Category D] (F : C ⥤ D) (X Y Z : C)
  (f : X ⟶ Y) (g : Y ⟶ Z) :
  F.map (f ≫ g) = F.map f ≫ F.map g := by
  exact F.map_comp f g

-- 练习3：自然变换的基本性质
-- 对应芝加哥大学范畴论标准
theorem naturality (C D : Type) [Category C] [Category D] (F G : C ⥤ D)
  (α : F ⟶ G) (X Y : C) (f : X ⟶ Y) :
  α.app Y ≫ G.map f = F.map f ≫ α.app X := by
  exact α.naturality f

-- 练习4：同构的基本性质
-- 对应华威大学范畴论标准
theorem iso_comp (C : Type) [Category C] (X Y Z : C) (f : X ≅ Y) (g : Y ≅ Z) :
  (f ≪≫ g).hom = f.hom ≫ g.hom := by
  rfl

-- 练习5：极限的基本性质
-- 对应巴黎第六大学范畴论标准
theorem limit_universal (C : Type) [Category C] (J : Type) [Category J] (F : J ⥤ C) :
  HasLimit F → ∃! (cone : Cone F), IsLimit cone := by
  intro h
  exact h.exists_unique

-- 练习6：伴随函子的基本性质
-- 对应伦敦大学学院范畴论标准
theorem adjunction_hom_equiv (C D : Type) [Category C] [Category D] (F : C ⥤ D) (G : D ⥤ C) :
  F ⊣ G → ∀ (X : C) (Y : D), (F.obj X ⟶ Y) ≃ (X ⟶ G.obj Y) := by
  intro h X Y
  exact h.homEquiv X Y

end CategoryTheoryExercises
