# The MonsterMatch Creative Tooling

This package was used to perform the simulated preferences of monsters inside [Monster Match](https://monstermatch.hiddenswitch.com).

## Installation

Requires Python 3.7

**macOS**: Install `libomp` first, the export variables to use it (requires `brew`).

```
brew install libomp
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
```


**All platforms**: Install the package in editable mode, then use the command line tool.

```bash
pip install -e .
monstermatch --help
```

## Theory

The core script generates [this file](https://github.com/hiddenswitch/Monster-Match/blob/master/Assets/Scripts/MonsterMatch/CollaborativeFiltering/MonsterMatchArrayData.cs) used by the recommendation [algorithm here](https://github.com/hiddenswitch/Monster-Match/blob/master/Assets/Scripts/MonsterMatch/CollaborativeFiltering/NonnegativeMatrixFactorization.cs) as its initial data.

Conceptually, this corresponds to the prior users' swipes. In this case, each monster is assumed to have swiped on every other monster at least once.

In the experiments to generate this data, it was conjectured that there was a certain user discrimination behavior (i.e., how to decide to swipe) that would correspond to the lived experience of users experiencing bias and discrimination on dating apps.

Specifically, is there a particular model for how users swipe that would create the result, "See a sequence of many of the same visible kind of monster" for the player?

This code conjectures that if a binary feature is used, and at least the majority of monsters make their decision to swipe right i.i.f. the binary features are the same, this behavior would be observed. This appears to be the case in the production version of Monster Match.

This experience was not observed if non-binary features were used or if recommendations were done in any other way besides collaborative filtering.