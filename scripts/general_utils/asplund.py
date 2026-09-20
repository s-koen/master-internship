import numpy as np
import periodictable as pt
import pickle as pkl


class Element:
    def __init__(self, Z: int, name: str, abundance: float) -> None:
        self.Z = Z
        self.name = name
        self.abundance = abundance
        self.atomic_mass = pt.elements[self.Z].mass
        self.relative_abundance = 10.0 ** (self.abundance - 12.0)

    def __repr__(self) -> str:
        return f"Element {self.Z}: log ϵ({self.name}) = {self.abundance})"

    def add_massfrac(self, massfrac: float) -> None:
        self.massfrac = massfrac


class Asplund:
    def __init__(self, z: float = 0.005573) -> None:
        self.elements: dict[int, Element] = {}
        self.z = z
        self.y = 0.24 + 2 * self.z
        self.x = 1 - self.y - self.z

        self.__add_all()
        self.__compute_massfracs()
        self.__scale_to_z()

    def __add(self, element: Element) -> None:
        self.elements[element.Z] = element

    def __add_all(self) -> None:
        self.__add(Element(1, "h", 12.00))
        self.__add(Element(2, "he", 10.93))
        self.__add(Element(3, "li", 1.05))
        self.__add(Element(4, "be", 1.38))
        self.__add(Element(5, "b", 2.70))
        self.__add(Element(6, "c", 8.43))
        self.__add(Element(7, "n", 7.83))
        self.__add(Element(8, "o", 8.69))
        self.__add(Element(9, "f", 4.56))
        self.__add(Element(10, "ne", 7.93))
        self.__add(Element(11, "na", 6.24))
        self.__add(Element(12, "mg", 7.60))
        self.__add(Element(13, "al", 6.45))
        self.__add(Element(14, "si", 7.51))
        self.__add(Element(15, "p", 5.41))
        self.__add(Element(16, "s", 7.12))
        self.__add(Element(17, "cl", 5.50))
        self.__add(Element(18, "ar", 6.40))
        self.__add(Element(19, "k", 5.03))
        self.__add(Element(20, "ca", 6.34))
        self.__add(Element(21, "sc", 3.15))
        self.__add(Element(22, "ti", 4.95))
        self.__add(Element(23, "v", 3.93))
        self.__add(Element(24, "cr", 5.64))
        self.__add(Element(25, "mn", 5.43))
        self.__add(Element(26, "fe", 7.50))
        self.__add(Element(27, "co", 4.99))
        self.__add(Element(28, "ni", 6.22))
        self.__add(Element(29, "cu", 4.19))
        self.__add(Element(30, "zn", 4.56))
        self.__add(Element(31, "ga", 3.04))
        self.__add(Element(32, "ge", 3.65))
        self.__add(Element(33, "as", 2.30))
        self.__add(Element(34, "se", 3.34))
        self.__add(Element(35, "br", 2.54))
        self.__add(Element(36, "kr", 3.25))
        self.__add(Element(37, "rb", 2.52))
        self.__add(Element(38, "sr", 2.87))
        self.__add(Element(39, "y", 2.21))
        self.__add(Element(40, "zr", 2.58))
        self.__add(Element(41, "nb", 1.46))
        self.__add(Element(42, "mo", 1.88))
        self.__add(Element(44, "ru", 1.75))
        self.__add(Element(45, "rh", 0.91))
        self.__add(Element(46, "pd", 1.57))
        self.__add(Element(47, "ag", 0.94))
        self.__add(Element(48, "cd", 1.71))
        self.__add(Element(49, "in", 0.80))
        self.__add(Element(50, "sn", 2.04))
        self.__add(Element(51, "sb", 1.01))
        self.__add(Element(52, "te", 2.18))
        self.__add(Element(53, "i", 1.55))
        self.__add(Element(54, "xe", 2.24))
        self.__add(Element(55, "cs", 1.08))
        self.__add(Element(56, "ba", 2.18))
        self.__add(Element(57, "la", 1.10))
        self.__add(Element(58, "ce", 1.58))
        self.__add(Element(59, "pr", 0.72))
        self.__add(Element(60, "nd", 1.42))
        self.__add(Element(62, "sm", 0.96))
        self.__add(Element(63, "eu", 0.52))
        self.__add(Element(64, "gd", 1.07))
        self.__add(Element(65, "tb", 0.30))
        self.__add(Element(66, "dy", 1.10))
        self.__add(Element(67, "ho", 0.48))
        self.__add(Element(68, "er", 0.92))
        self.__add(Element(69, "tm", 0.10))
        self.__add(Element(70, "yb", 0.84))
        self.__add(Element(71, "lu", 0.10))
        self.__add(Element(72, "hf", 0.85))
        self.__add(Element(73, "ta", -0.12))
        self.__add(Element(74, "w", 0.85))
        self.__add(Element(75, "re", 0.26))
        self.__add(Element(76, "os", 1.40))
        self.__add(Element(77, "ir", 1.38))
        self.__add(Element(78, "pt", 1.62))
        self.__add(Element(79, "au", 0.92))
        self.__add(Element(80, "hg", 1.17))
        self.__add(Element(81, "tl", 0.90))
        self.__add(Element(82, "pb", 1.75))
        self.__add(Element(83, "bi", 0.65))
        self.__add(Element(90, "th", 0.02))
        self.__add(Element(92, "u", -0.54))

    def __compute_massfracs(self) -> None:
        massfracs: list[float] = []
        for Z, element in self.elements.items():
            massfracs.append(element.relative_abundance * element.atomic_mass)

        massfracs = list(np.array(massfracs) / np.sum(massfracs))
        for i, (Z, element) in enumerate(self.elements.items()):
            element.add_massfrac(massfracs[i])

    def __scale_to_z(self) -> None:

        original_x = self.elements[1].massfrac
        original_y = self.elements[2].massfrac
        original_z = 1 - original_x - original_y

        for Z, element in self.elements.items():
            if Z == 1:
                element.massfrac = self.x
                continue

            if Z == 2:
                element.massfrac = self.y
                continue

            element.massfrac *= self.z / original_z
