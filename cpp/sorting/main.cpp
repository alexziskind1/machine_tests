//
//  main.cpp
//  cpptest
//
//  Created by Alexander Ziskind on 1/11/21.
//

#include <iostream>

#include<stdio.h>

#include <cstdlib>
#include <ctime>


  
// A utility function to swap two elements
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
int partition (int arr[], int low, int high)
{
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
  
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
  
/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
  
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
  
/* Function to print an array */
void printArray(int arr[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", arr[i]);
    printf("n");
}
  
// Driver program to test above functions
int main()
{
    int sz;
    sz = 10000000;
    int* arr = new int[sz];

    srand(time(nullptr)); // Initialize random seed

    for(int i=0;i<sz;i++)
        arr[i]=rand()%100;  //Generate number between 0 to 99
    
    int n = sz;
    quickSort(arr, 0, n-1);
    printf("Sorted array: n");
    printArray(arr, n);
    delete[] arr;
    return 0;
}