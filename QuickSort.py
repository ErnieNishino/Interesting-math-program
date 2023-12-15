class QuickSort:
    def _find_mid(self, nums: list[int], left: int, mid: int, right: int):
        """
        return the middle one among left, mid, right
        """
        if (nums[left] < nums[mid]) ^ (nums[left] < right):
            return left
        elif (nums[right] < nums[mid]) ^ (nums[right] < nums[left]):
            return right
        else:
            return mid

    def _partition(self, nums: list[int], left: int, right: int):
        """
        find the partition
        """
        mid = self._find_mid(nums, left, (left + right) // 2, right)
        nums[left], nums[mid] = nums[mid], nums[left]

        i, j = left, right
        while i < j:
            while i < j and nums[j] >= nums[left]:
                j -= 1
            while i < j and nums[i] <= nums[left]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i], nums[left] = nums[left], nums[i]

        return i

    def quick_sort(self, nums: list[int], left: int, right: int):
        if left < right:
            pivot = self._partition(nums, left, right)
            self.quick_sort(nums, left, pivot - 1)
            self.quick_sort(nums, pivot + 1, right)


if __name__ == "__main__":
    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    QuickSort().quick_sort(nums, 0, len(nums) - 1)
    print(nums)
