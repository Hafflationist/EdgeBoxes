from typing import Dict, Set, TypeVar


T = TypeVar("T")


class IdDict(Dict[T, T]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key: T) -> T:
        self[key] = key
        return key


class IdDictInverse(Dict[T, Set[T]]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key: T) -> Set[T]:
        self[key] = {key}
        return {key}
