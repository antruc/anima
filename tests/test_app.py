from depuradordecamaratrampa.app import should_copy


def test_should_copy():
    assert should_copy("a.jpg", "", 2)
    assert not should_copy("a.jpg", "", 0)
    # Video frames repeat the same path, copy only once
    assert not should_copy("v.mp4", "v.mp4", 3)
    assert should_copy("b.jpg", "a.jpg", 1)
