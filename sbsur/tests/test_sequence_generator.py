from sbsur import SequenceGenerator


def test_with_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: [[1]], 2, 0)
        assert not seq_gen.is_exhausted()
    except:
        assert False


def test_with_none_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: [[1]], 2, None)
        assert not seq_gen.is_exhausted()
    except:
        assert False