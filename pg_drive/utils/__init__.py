def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


def recursive_equal(data1, data2, need_assert=False):
    if isinstance(data1, dict):
        assert isinstance(data2, dict)
        assert set(data1.keys()) == set(data2.keys()), (data1.keys(), data2.keys())
        ret = []
        for k in data1:
            ret.append(recursive_equal(data1[k], data2[k]))
        return all(ret)

    elif isinstance(data1, list):
        assert len(data1) == len(data2)
        ret = []
        for i in range(len(data1)):
            ret.append(recursive_equal(data1[i], data2[i]))
        return all(ret)

    else:
        ret = data1 == data2
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret
