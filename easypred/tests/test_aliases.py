def test_no_error_when_direct_import():
    try:
        import easypred.type_aliases
    except Exception as e:
        assert False, f"Type aliases import caused an exception: {e}"
