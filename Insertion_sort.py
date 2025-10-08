def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Best case: array already sorted, so inner loop doesn't run
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Best-case input: already sorted
arr = [1, 2, 3, 4, 5, 6]
sorted_arr = insertion_sort(arr)
print("Sorted array:", sorted_arr)
