import pytest

from src.chunking import chunk_text

def test_chunking_without_overlap():
    text = 'a b c d e f g h'
    expected = ['a b', 'c d', 'e f', 'g h']
    res = chunk_text(text, chunk_size=2, overlap=0)

    assert res == expected


@pytest.mark.parametrize(
    'chunk_size, overlap',
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text('a b c', chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize('text', ['', '   \t\n'])
def test_empty_text(text):
    assert chunk_text(text, chunk_size=4, overlap=1) == []

def test_chunking_with_overlap():
    text = 'a b c d e f'
    expected = ['a b c d', 'c d e f']
    res = chunk_text(text, chunk_size=4, overlap=2)

    assert res == expected

def test_chunking_last_chunk():
    text = 'a b c d e'
    expected = ['a b c d', 'e']

    res = chunk_text(text, chunk_size=4, overlap=0)

    assert res == expected
