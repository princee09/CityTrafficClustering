class Solution {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;

        while (left < right) {
            char leftChar = s.charAt(left);
            char rightChar = s.charAt(right);

            // Step 1: Skip non-alphanumeric characters from the left
            if (!Character.isLetterOrDigit(leftChar)) {
                left++;
            } 
            // Step 2: Skip non-alphanumeric characters from the right
            else if (!Character.isLetterOrDigit(rightChar)) {
                right--;
            } 
            // Step 3: Both are alphanumeric, compare them
            else {
                if (Character.toLowerCase(leftChar) != Character.toLowerCase(rightChar)) {
                    return false;
                }
                left++;
                right--;
            }
        }
        
        return true;
    }
}