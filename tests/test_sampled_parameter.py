import numpy as np
from galibrate.sampled_parameter import SampledParameter


def test_init():
    assert SampledParameter("test", 0.0, 1.0) is not None


def test_attributes():
    sp = SampledParameter("test", 0.0, 1.0)
    assert sp.name == "test"
    assert np.isclose(sp.loc, 0.0)
    assert np.isclose(sp.width, 1.0)


def test_member_random():
    sp = SampledParameter("test", 0.0, 1.0)
    rand = sp.random(1)
    assert len(rand) == 1
    assert rand[0] < (sp.loc + sp.width)
    assert rand[0] > (sp.loc)


def test_member_unit_transform():
    sp = SampledParameter("test", 0.0, 1.0)
    utrans = sp.unit_transform(0.0)
    assert np.isclose(utrans, sp.loc)
    utrans = sp.unit_transform(1.0)
    assert np.isclose(utrans, (sp.loc + sp.width))


if __name__ == "__main__":
    test_init()
    test_attributes()
    test_member_random()
    test_member_unit_transform()
