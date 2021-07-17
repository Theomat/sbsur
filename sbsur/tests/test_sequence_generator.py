from sbsur import SequenceGenerator


def test_with_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: [1], 2, 0)
    except:
        assert False


def test_with_none_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: [1], 2, None)
    except:
        assert False