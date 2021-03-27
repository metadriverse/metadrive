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
        c.update({"aa": {"bb": 102}})
    except TypeError:
        pass
    else:
        raise ValueError()

    try:
        c.update({"aa": {"bbd": 102}})
    except KeyError:
        pass
    else:
        raise ValueError()

    c.update({"aa": {"bbd": 103}}, allow_overwrite=True)
    assert c.aa.bbd == 103
    assert c.aa.bb.cc == 101

    c.update({"aa": {"bb": 102}}, allow_overwrite=True)
    assert c.aa.bb == 102


def test_partially_update():
    c = PGConfig({"aa": {"bb": {"cc": 100}}})

    try:
        c.update({"aa": {"bb": {"dd": 101}}}, allow_overwrite=False)
    except KeyError:
        pass
    else:
        raise ValueError()

    c.update({"aa": {"bb": {"dd": 101}}}, allow_overwrite=True)
    assert c.aa.bb.cc == 100
    assert c.aa.bb.dd == 101


if __name__ == '__main__':
    test_recursive_config()
    test_partially_update()
