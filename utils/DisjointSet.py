from collections import defaultdict
from typing import Any, DefaultDict, Generic, Iterator, Set, Tuple, TypeVar

from utils.IdDict import IdDict, IdDictInverse

T = TypeVar("T")


class DisjointSet(Generic[T]):

    def __init__(self, *args, **kwargs) -> None:
        self._data: IdDict[T] = IdDict(*args, **kwargs)
        self._data_inverse: IdDictInverse[T] = IdDictInverse(*args, **kwargs)

    def __contains__(self, item: T) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)

    def __get__(self, element: T) -> T:
        return self.find(element)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DisjointSet):
            return False

        return {tuple(x) for x in self.iter_sets()} == {tuple(x) for x in other.iter_sets()}

    def __repr__(self) -> str:
        sets = {key: val for key, val in self}
        return f"{self.__class__.__name__}({sets})"

    def __str__(self) -> str:
        return "{classname}({values})".format(
            classname=self.__class__.__name__, values=", ".join(str(dset) for dset in self.iter_sets()),
        )

    def __iter__(self) -> Iterator[Tuple[T, T]]:
        for key in self._data.keys():
            yield key, self.find(key)

    def iter_sets(self) -> Iterator[Set[T]]:
        element_classes: DefaultDict[T, Set[T]] = defaultdict(set)
        for element in self._data:
            element_classes[self.find(element)].add(element)
        yield from element_classes.values()

    def iter_sets_with_canonical_elements(self) -> Iterator[Tuple[T, Set[T]]]:
        element_classes: DefaultDict[T, Set[T]] = defaultdict(set)
        for element in self._data:
            element_classes[self.find(element)].add(element)
        yield from element_classes.items()

    def iter_specific_set(self, value: T) -> Set[T]:
        canonical_value = self.find(value)
        return self._data_inverse[canonical_value]

    def find(self, x: T) -> T:
        while x != self._data[x]:
            self._data[x] = self._data[self._data[x]]
            x = self._data[x]
        return x

    def union(self, x: T, y: T) -> None:
        parent_x, parent_y = self.find(x), self.find(y)
        if parent_x != parent_y:
            self._data_inverse[parent_y] = self._data_inverse[parent_y] | self._data_inverse[parent_x]
            self._data[parent_x] = parent_y

    def connected(self, x: T, y: T) -> bool:
        return self.find(x) == self.find(y)
