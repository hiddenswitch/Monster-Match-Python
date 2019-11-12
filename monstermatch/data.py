import dataclasses
import enum
from collections import Counter
from textwrap import dedent
from typing import List, Optional, Union, Tuple, Mapping
from itertools import chain

import numpy as np
from numpy import ma
from sklearn.decomposition import NMF

_zero_index_monster_type_column = [
    'Humanoid',
    'Insect',
    'Humanoid',
    'Creature',
    'Mixed',
    'Humanoid',
    'Humanoid',
    'Humanoid',
    'Humanoid',
    'Insect',
    'Humanoid',
    'Mixed',
    'Humanoid',
    'Creature',
    'Insect',
    'Humanoid',
    'Humanoid',
    'Creature',
    'Humanoid',
    'Mech',
    'Mixed',
    'Mixed',
    'Creature',
    'Mixed',
    'Mech',
    'Creature',
    'Humanoid',
    'Creature',
    'Mech',
    'Mixed',
    'Humanoid',
    'Mech',
    'Humanoid',
    'Humanoid',
    'Humanoid',
    'Mech',
    'Humanoid',
    'Humanoid',
    'Mech',
    'Humanoid',
    'Insect',
    'Humanoid',
    'Humanoid',
    'Mech',
    'Mixed',
    'Humanoid',
    'Mech',
    'Creature',
    'Humanoid',
    'Mech',
    'Creature',
    'Humanoid',
    'Humanoid',
    'Mixed',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Mech',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature',
    'Creature'
]


def _get_data():
    return _zero_index_monster_type_column[:]


class MonsterType(enum.Enum):
    HUMANOID = 'Humanoid'
    CREATURE = 'Creature'
    MECH = 'Mech'
    INSECT = 'Insect'
    MIXED = 'Mixed'


@dataclasses.dataclass
class Rating:
    user_id: int
    item_id: int
    rating: float


@dataclasses.dataclass(frozen=True)
class MonsterProfile:
    index: int
    monster_type: MonsterType


def to_typed_list(monster_types: List[str], supported_monster_types=(MonsterType.HUMANOID, MonsterType.CREATURE)) -> \
        List[MonsterProfile]:
    typed_list = []  # type: List[MonsterProfile]
    for i, _monster_type in enumerate(monster_types):
        monster_type = MonsterType[_monster_type.upper()]
        if monster_type in supported_monster_types:
            typed_list += [MonsterProfile(i, monster_type)]

    return typed_list


def remap(in_data: List[MonsterProfile], length: int) -> List[Optional[MonsterProfile]]:
    _data = [None] * length
    for item in in_data:
        _data[item.index] = item
    return _data


def data() -> List[MonsterProfile]:
    return to_typed_list(_get_data())


def masked_array_from_ratings(vals: Union[List[Rating], List[Tuple[int, int, float]]], rows: int,
                              columns: int) -> ma.MaskedArray:
    try:
        vals = [dataclasses.astuple(a) for a in vals]
    except:
        pass
    base_array = np.zeros(shape=(rows, columns))
    mask_array = np.ones(shape=(rows, columns))
    for (i, j, v) in vals:
        base_array[i, j] = v
        mask_array[i, j] = 0
    return ma.MaskedArray(data=base_array, mask=mask_array)


def nan_masked_from_ratings(vals: Union[List[Rating], List[Tuple[int, int, float]]], rows: int,
                            columns: int) -> np.ndarray:
    try:
        vals = [dataclasses.astuple(a) for a in vals]
    except:
        pass
    base_array = np.empty(shape=(rows, columns))
    base_array[:] = np.nan
    for (i, j, v) in vals:
        base_array[i, j] = v
    return base_array


def generate_w_h(latents: List[MonsterProfile], n_users: int = 100,
                 n_random_components: int = 1, use_random:bool=True, sigma: float = 1, mean: float = -4) -> Tuple[
    np.ndarray, np.ndarray, Mapping[int, MonsterType]]:
    latent_counts = Counter(x.monster_type for x in latents)  # type: Mapping[MonsterType, int]
    mapping = dict([(i, latent) for (i, latent) in enumerate(latent_counts.keys())])
    n_components = len(mapping) + n_random_components
    n_items = len(latents)
    W = np.zeros(shape=(n_users + 1, n_components))
    H = np.zeros(shape=(n_components, n_items))
    # The preference is 1.0 if it's the same latent, 0.0 if it's different
    user_latents = []  # type: List[MonsterType]
    for latent, count in latent_counts.items():
        user_latents += [latent] * round(count / len(latents) * n_users)

    for i, latent_i in enumerate(user_latents):
        for k, latent_k in mapping.items():
            W[i, k] = 1.0 if latent_i == latent_k else 0.0

    for k, latent_k in mapping.items():
        for j, latent_j in enumerate(latents):
            H[k, j] = 1.0 if latent_k == latent_j.monster_type else 0.0

    if use_random:
        W[:, -n_random_components:] = np.random.lognormal(mean=mean, sigma=sigma,
                                                          size=(n_users + 1, n_random_components))
        H[-n_random_components:, :] = np.random.lognormal(mean=mean, sigma=sigma, size=(n_random_components, n_items))
    W[n_users, :] = 1 / n_components + np.random.lognormal(mean=mean, sigma=sigma, size=(n_components,))
    return W, H, mapping


def cshar_repr_double(j) -> str:
    if np.isnan(j):
        return 'double.NaN'
    else:
        return str(j)


def csharp_repr_ndarray(in_array: np.ndarray) -> str:
    return 'new[,] {' + ','.join(
        ['{' + ','.join([cshar_repr_double(j) for j in in_array[i]]) + '}' for i in range(in_array.shape[0])]) + '}'


def generate_array_data_file():
    latents = data()
    # W = np.zeros(shape=(n_users + 1, n_components))
    # H = np.zeros(shape=(n_components, n_items))
    n_users = 100
    W, H, mapping = generate_w_h(latents, n_users=n_users, use_random=True, sigma=.5, mean=-1.5)
    nmf = NMF(solver='mu', init='custom', n_components=3)
    nmf.components_ = H
    nmf.n_components_ = H.shape[0]
    X = nmf.inverse_transform(W)
    # returns a shape of n_users+1. Wipe these ratings since they are not yet determined.
    X[n_users, :] = np.nan
    file_contents = dedent('''
namespace MonsterMatch.CollaborativeFiltering
{
    public static class MonsterMatchArrayData
    {
        // @formatter:off
        public const int UserCount = %d;
        public const int ItemCount = %d;
        public const int PlayerUserId = %d;
        public const int FactorCount = %d;
        public static readonly int[] ForProfiles = %s;
        public static readonly double[,] Data = %s;
        public static readonly double[,] Pu = %s;
        public static readonly double[,] Qi = %s;
        // @formatter:on
    }
}
    ''')
    file_contents = file_contents % (
        n_users+1,
        len(latents),
        n_users,
        nmf.n_components_,
        'new [] {'+','.join([str(profile.index) for profile in latents])+'}',
        csharp_repr_ndarray(X),
        csharp_repr_ndarray(W),
        csharp_repr_ndarray(H.T)
    )
    print(file_contents)
