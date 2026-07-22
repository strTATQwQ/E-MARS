from scripts.convert_mp3d_to_usd import has_nonzero_area_triangle


class Attr:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value


class Mesh:
    def __init__(self, points, counts, indices):
        self.points, self.counts, self.indices = points, counts, indices

    def GetPointsAttr(self):
        return Attr(self.points)

    def GetFaceVertexCountsAttr(self):
        return Attr(self.counts)

    def GetFaceVertexIndicesAttr(self):
        return Attr(self.indices)


def test_nonzero_area_filter_rejects_collinear_mesh():
    assert not has_nonzero_area_triangle(Mesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [3], [0, 1, 2]))


def test_nonzero_area_filter_accepts_triangle():
    assert has_nonzero_area_triangle(Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [3], [0, 1, 2]))
