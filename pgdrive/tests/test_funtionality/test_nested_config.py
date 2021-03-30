from pgdrive.utils.pg_config import PGConfig


def test_recursive_config():
    c = PGConfig({"aa": {"bb": {"cc": 100}}})
    assert c.aa.bb.cc == 100
    assert isinstance(c.aa, PGConfig)
    assert isinstance(c.aa.bb, PGConfig)
    assert isinstance(c.aa.bb.cc, int)

    c.update({"aa": {"bb": {"cc": 101}}})
    assert c.aa.bb.cc == 101

    try:
        c.update({"aa": {"bb": 102}}, allow_overwrite=False)
    except TypeError:
        pass
    else:
        raise ValueError()

    try:
        c.update({"aa": {"bbd": 102}}, allow_overwrite=False)
    except KeyError:
        pass
    else:
        raise ValueError()

    c.update({"aa": {"bbd": 103}})
    assert c.aa.bbd == 103
    assert c.aa.bb.cc == 101

    c.update({"aa": {"bb": 102}})
    assert c.aa.bb == 102


def test_partially_update():
    c = PGConfig({"aa": {"bb": {"cc": 100}}})

    try:
        c.update({"aa": {"bb": {"dd": 101}}}, allow_overwrite=False)
    except KeyError:
        pass
    else:
        raise ValueError()

    c.update({"aa": {"bb": {"dd": 101}}})
    assert c.aa.bb.cc == 100
    assert c.aa.bb.dd == 101


def test_config_identical():
    c = PGConfig({"aa": {"bb": {"cc": 100}}})
    d = PGConfig({"aa": {"bb": {"cc": 100}}})
    assert c.is_identical(c)
    assert c.is_identical(d)
    assert d.is_identical(c)
    assert d.is_identical(d)

    c.aa.bb.cc = 101
    assert c.is_identical(c)
    assert d.is_identical(d)
    assert not c.is_identical(d)
    assert not d.is_identical(c)

    c.update({"aa": {"bb": 10001}})
    assert c.is_identical(c)
    assert d.is_identical(d)
    assert not c.is_identical(d)
    assert not d.is_identical(c)


if __name__ == '__main__':
    test_recursive_config()
    test_partially_update()
    test_config_identical()
