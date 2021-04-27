from pgdrive.utils import PGConfig


def test_config_unchangeable():
    c = PGConfig({"aaa": 100}, unchangeable=True)
    try:
        c['aaa'] = 1000
    except ValueError as e:
        print('Great! ', e)
    assert c['aaa'] == 100


if __name__ == '__main__':
    test_config_unchangeable()
