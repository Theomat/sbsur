from sbsur import SequenceGenerator


def test_with_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: None, 2, 0)
    except:
        assert False


def test_with_nonee_seed():
    try:
        seq_gen = SequenceGenerator(lambda _: None, 2, None)
    except:
        assert False