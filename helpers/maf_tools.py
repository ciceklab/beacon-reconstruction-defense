import torch

class MAFHelper:

    def __init__(self, mafs) -> None:
        self.mafs = mafs.clone().detach()
        if self.mafs.dim() > 1:
            self.mafs = self.mafs.view(-1)
        self.maf_categories = self._get_categorized_mafs_vectorized()

    def get_mafs(self) -> torch.Tensor:
        return self.mafs.clone()

    def get_mafs_categories(self) -> torch.Tensor:
        return self.maf_categories.clone()

    def _get_categorized_mafs_vectorized(self):
        boundaries = torch.tensor([0.01, 0.05, 0.1, 0.3], device=self.mafs.device)
        cats = torch.bucketize(self.mafs, boundaries)
        return cats

    def get_category_static(self, maf) -> int:
        if maf <= 0.01:
            return 0
        elif maf <= 0.05:
            return 1
        elif maf <= 0.1:
            return 2
        elif maf <= 0.3:
            return 3
        else:
            return 4
