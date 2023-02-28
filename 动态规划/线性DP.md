<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 线性DP </font> </center>

### 剑指 Offer 10- I. 斐波那契数列
题意：给定n，求斐波那契数列的第n项

#### 思路1：递归(时间复杂度高，容易超时)
```c++
class Solution {
public:
    int fib(int n) {
        return n <= 1 ? n : fib(n - 1) + fib(n - 2);
    }
};
```

#### 思路2：动态规划
**时间复杂度$O(n)$   空间复杂度$O(n)$**
```c++
class Solution {
public:
    int fib(int n) {
        if (n < 2) return n;
        vector<int>f(n + 1);
        f[0] = 0, f[1] = 1;
        for (int i = 2; i <= n; i++) f[i] = f[i - 1] + f[i - 2];
        return f[n];
    }
};
```

**用滚动数组(变量)优化**
**时间复杂度$O(n)$   空间复杂度$O(1)$**
```c++
class Solution {
public:
    int fib(int n) {
        if (n < 2) return n;
        int a = 0, b = 1, c;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        } 
        return c;
    }
};
```

#### 思路3：矩阵快速幂
<img src=../Fig/矩阵快速幂.PNG>


**时间复杂度$O(logn)$** 
```c++
class Solution {
public:
    typedef vector<vector<int>> vvi;
    // 适用于任何合法矩阵的乘法
    vvi mul (vvi& a, vvi& b) {
        vvi c = vvi(a.size(), vector<int>(b[0].size(), 0)); // 定义大小
        for (int i = 0; i < a.size(); i++) 
            for(int j = 0; j < b[0].size(); j++) 
                for(int k = 0; k < a[0].size(); k++) 
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }
    int fib(int n) {
        if (n < 2) return n;

        vvi x ={{1, 0}};
        vvi res = {{1, 0}, {0, 1}}; //单位矩阵
        vvi A = {{1, 1},{1, 0}}; 
        int b = n - 1;  //求A^(n-1)
        for (; b; b >>= 1) {
            if(b & 1) res = mul(res, A);
            A = mul(A, A); 
        }
        vvi ans = mul(x, res);
        return ans[0][0];
    }
};
```
---

### 62. 不同路径
题意：给定一个$m \times n$的矩形，问从左上角到右下角有多少种走法，每次只能向右或者向下。

#### 思路1：DP

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>>f(m + 1, vector<int>(n + 1, 0));
        f[0][1] = 1;
        for (int i = 1; i <= m; i++) 
            for (int j = 1; j <= n; j++) 
                f[i][j] = f[i - 1][j] + f[i][j - 1]; 
        return f[m][n];   
    }
};
```
**压缩到一维**
```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int>f(n, 0);
        f[0] = 1;
        for (int i = 0; i < m; i++) 
            for (int j = 0; j < n; j++) 
                if(j > 0) f[j] += f[j - 1];
        return f[n - 1];   
    }
};
```

#### 思路2：数学
$m \times n$的矩形中从左上走到右下一共有`m + n - 2`步, 每一步可以选择`right(0)` 或者 `down(1)`, 因此八个位置`00000011`选择 m - 1个向下就符合题意，因此答案就是求$C^{m-1}_{m+n-2} = C^{n-1}_{m+n-2} = \frac{(m+n-2)(m+n-1)\dots m}{(n-1)!}$
分子最后为`m`的原因是因为分子分母都是`n-1`项
```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        long long  ans = 1; 
        for(int i = m, j = 1; j <= n - 1; i++, j++) 
            ans = ans * i / j;
        return ans;
    }
};
```
这样计算为啥能保证每次相乘都是整数呢？
因为这样算的话：第一次结果等于$C_{m}^{1}$,后面依次是：$C_{m+1}^{2}$, $C_{m+2}^{3} \dots$每次都是组合数，而组合数为整数。

---

### 63. 不同路径 II
题意：在不同路径的基础上加了障碍物

```c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        vector<int>dp(m, 0);
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if(a[i][j] == 1) dp[j] = 0;
                else if (j > 0) dp[j] +=  dp[j - 1];
            }
        }
        return dp[m - 1];
    }
};
```
---

### 343. 整数拆分 或 剑指 Offer 14- II. 剪绳子 
题意：将一个数字拆分为至少两份, 求最大的乘积
`5 = 2 * 3 = 6`

#### 思路1：DP
`dp[i]` 表示将正整数 `i` 拆分成至少两个正整数的和之后，这些正整数的最大乘积。
需要注意`dp[i]`不一定大于`i` 例如`dp[3] = 2 < 3`
因此状态转移`dp[i] = max(dp[j] * dp[i - j] for j in range(i))`是错的
正确应该为：`dp[i] = max(max(dp[j], j) * max(dp[i - j], i - j))`
分成的两部分不确定到底是`dp[x]`大还是`x`大，就选择两者较大的

**时间复杂度$O(n^2)$**
```c++
class Solution {
public:
    int integerBreak(int n) {
        vector<int>dp(n + 1);
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                int num = max(j, dp[j]) * max(i - j,dp[i - j]);
                dp[i] = max(dp[i], num);
            }
        }
        return dp[n];
    }
};
```

#### DP的优化：
通过简单的计算可以发现:
`dp[2] = 1 < 2, dp[3] = 2 < 3, dp[4] = 4 = 4, dp[5] = 6 > 5`
只有`dp[2]`和`dp[3]`不拆分比较好，剩余的数拆分比较好。

**时间复杂度$O(n)$**
```c++
class Solution {
public:
    int integerBreak(int n) {
        vector<int>f(60);
        f[1] = 1, f[2] = 1, f[3] = 2;
        for(int i = 4; i <= n; i++) 
            f[i] = max(max(2 * f[i - 2], 2 * (i - 2)), max(3 * f[i - 3], 3 * (i - 3)));
        return f[n];
    }
};
```

#### 思路2：数学
结论：
- 如果`n%3==0` 全部拆分成3
- 如果`n%3==1` 拆成2个2，剩下的全为3
- 如果`n%3==2` 拆成1个2，剩下的全为3

简单证明：
$n=n_1+n_2+\dots +n_k$
1. 假设某个$n_i>5$, 则可以把$n_i$继续拆分为3，$n_i-3$这两个数的乘积$3(n_i-3)>n_i$  推导：$3n_i-9>n_i$   ->   $2n_i>9$  显然成立。所以如果存在大于5的数，只会导致乘积变小。
2. 假设$n_i==4$, 则可以将其分成2*2，可以不包含4
3. 假设存在$n_i=1$,乘1不会使得乘积增大，反而会使得乘积变小
4. 所以此时只能拆分成2和3。
5. 假设存在3个2，`2*2*2<3*3`,所以最多只有两个2

**时间复杂度$O(1)$**
```c++
class Solution {
public:
    int integerBreak(int n) {
        if (n <= 3) return n - 1;
        int p = n / 3;
        int r = n % 3;
        if (r == 0) return pow(3, p);
        else if(r == 1) return pow(3, p - 1) * 4;
        else return pow(3, p) * 2;
    }
};
```
`pow`函数比较慢
```C++
class Solution {
public:
    int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        int ans = 1;
        if(n % 3 == 1) n -= 4, ans = 4;
        else if(n % 3 == 2) n-= 2, ans = 2;
        while(n) {
            n -= 3;
            ans *= 3;
        } 
        return ans;
    }
};
```

---

### 96. 不同的二叉搜索树
题意：给定一个数n，返回由`1-n`节点组成的二叉搜索数的个数
二叉搜索数的定义，根节点左边的节点都小于根节点，根节点右边的节点都大于根节点。
`n=3, ans = 5`
<img src=../Fig/二叉搜索树.PNG>

#### 思路1：DP
在求`n`时,分别考虑`x=1~n`分别作为`root`的时候的情况。

**时间复杂度$O(n^2)$**
```c++
class Solution {
public:
    int numTrees(int n) {
        if(n < 2) return n;
        vector<int>f(n + 1);
        f[0] = 1, f[1] = 1;
        for (int i = 2; i <= n; i++) 
            for (int j = 1; j <= i; j++) // j做root
                f[i] += f[j - 1] * f[i - j];
        return f[n];
    }
};
```

#### 思路2： 卡特兰数
<img src = ../Fig/卡特兰数1.PNG>
<img src = ../Fig/卡特兰数2.PNG>

其实上题动态规划的思路所引出的公式也就卡特兰数，所有的有关卡特兰数的题也可以这样想。
即：`f(4) = f(0)f(3) + f(1)f(2) + f(2)f(1) + f(3)f(0)`
如果发现题目有这样的性质，直接想卡特兰数
$f(n) = \sum_{i=1}^{n} f(i-1)f(n-i)$

**用公式(2)**
```c++
class Solution {
public:
    int numTrees(int n) {
        long long ans = 1;
        for(int i = 1, j = n + 1; i <= n; i++, j++) 
            ans = ans * j / i;
        ans /= (n + 1);
        return ans;
    }
};
```
**用公式(3)**
```c++
class Solution {
public:
    int numTrees(int n) {
        long long ans = 1;
        for (int i = 1; i <= n; i++) 
            ans = ans * (4 * i - 2) / (i + 1);
        return ans;
        
    }
};
```
---

### 1049. 最后一块石头的重量 II
题意： 将数组中的元素分成两个集合，使得两个**集合内所有元素的和**的差值尽可能小
`stones = [2,7,4,1,8,1], ans = 1`

#### 思路：转化为01背包
```c++
class Solution {
public:
    int lastStoneWeightII(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        int v = sum / 2;
        vector<int>f(v + 1, 0);
        for (auto x : nums) 
            for (int j = v; j >= x; j--)
                f[j] = max(f[j], f[j - x] + x);
        return sum - 2 * f[v];
    }
};
```
---

### 494. 目标和
题意：给一个数组，每个位置的元素有两种选择，其本身或者相反数，但所有位置都必须选，不能跳过，问组成值为`target`的方案数

#### 思路1：DFS
因为每个元素有两种选择，所以可以枚举所有的情况
时间复杂度：$O(2^n)$

```c++
class Solution {
public:
    int res = 0;
    void DFS(vector<int>& nums, int x, int k) {
        if(k == nums.size()) {
            if (!x) res++;
            return;
        }
        DFS(nums, x - nums[k], k + 1); // 选nums[k]
        DFS(nums, x + nums[k], k + 1); // 选-nums[k]
    } 
    int findTargetSumWays(vector<int>& nums, int target) {
        DFS(nums, target, 0);
        return res;
    }
};
```
**简化版**
```c++
class Solution {
public:
    
    int DFS(vector<int>& nums, int x, int k) {
        if(k == nums.size()) return !x;
        return DFS(nums, x - nums[k], k + 1) + DFS(nums, x + nums[k], k + 1);
    } 
    int findTargetSumWays(vector<int>& nums, int target) {
        return DFS(nums, target, 0);
    }
};
```

#### 思路2：DP
假设分成了两个集合，`A,B` `A`集合元素保持原数，`B`集合元素取反
因此有：`A + B = sum, A - B = target`
因此：`A = (sum + target) / 2`,其中`sum`和`target`都是已知的
因此问题就转化为在原数组中选几个数组成背包体积为A的方案数。
```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if((sum + target) % 2) return 0;
        if(sum + target < 0) return 0;
        int v = (sum + target) / 2;
        vector<int>f(v + 1, 0);
        f[0] = 1;
        for (auto x : nums) 
            for (int j = v; j >= x; j--) 
                f[j] += f[j - x];
        return f[v];
    }
};
```
---

### 518. 零钱兑换 II
题意：用数组中的某些数组成某个数`target`, 每个数可以无限次使用，问有多少种组合方案。
`target = 5, coins = [1, 2, 5], ans = 4`

#### 思路：完全背包
```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int>f(amount + 1, 0);
        f[0] = 1;
        for (auto x : coins) 
            for (int j = x; j <= amount; j++) 
                f[j] += f[j - x];
        return f[amount];
    }
};
```
---

### 377. 组合总和 Ⅳ
题意：用**不包含重复元素的数组**中的某些数组成某个数`target`, 每个数可以无限次使用，问有多少种组合方案。不同的顺序视为不同的方案，
例如`[1,2,1]和[2,1,1]`算两种方案。

`target = 5, coins = [1, 2, 5], ans = 9`

#### 思路：DP
状态表示:`f[i]`: 表示由数组中的元素组成值为i的组合数
状态转移:`f[i] += f[i - x] for x in nums`
即考虑最后一个元素为合法的`x`,前面的和为`i-x`的个数
```c++
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned long long>f(target + 1);
        f[0] = 1;
        for (int j = 1; j <= target; j++) 
            for (auto x : nums) 
                if (j >= x) f[j] += f[j - x];
        return f[target];
    }
};
```
**另一种理解：**
如果求组合数就是外层遍历物品，内层遍历体积。
如果求排列数就是外层遍历体积，内层遍历物品。

**进阶问题：如果原数组中有负数会出现什么情况？**
如果给定的数组中含有负数，则会导致出现无限长度的排列。
假设数组中有`a`和`-b`，那么只要选`a`个`-b`，`b`个`a`就可以构造出0.因此必须限制排列的长度。

---
### 322. 零钱兑换
题意：在数组中选尽可能少的数组成目标值`target`, 每个数可以重复选
`coins = [1, 2, 5], target = 11, ans = 3 [11 = 5 + 5 + 1]`
```c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int>f(amount + 1, 1e9);
        f[0] = 0;
        for (auto x : coins)    
            for (int j = x; j <= amount; j++) 
                f[j] = min(f[j], f[j - x] + 1);
        return f[amount] == 1e9 ? - 1: f[amount];
    }
};
```
---

### 139. 单词拆分
题意：问一个给定的字符串是否可以由字符串列表中的字符拼接而成，列表中的字符串可以重复使用
`s = "applepenapple", wordDict = ["apple", "pen"], ans = True`

#### 思路：完全背包[求排列]
外层遍历体积，内层遍历物品

更深层次的理解：
状态表示：`dp[j]`：表示字符串`s`的前`j`位(包括`j`)是否可以表示
状态转移：`dp[j] = dp[j] | dp[j - x.size()] for x in wordDict if s[-m] == x` 

```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<int>f(n + 1, 0);
        f[0] = 1;
        for (int j = 1; j <= n; j++) // 枚举体积
            for (auto e : wordDict) { // 枚举物品
                int m = e.size();
                if (j >= m) {
                    string ss = s.substr(j - m + 1 - 1, m); // - 1的原因是因为s的索引从0开始
                    if (ss == e) f[j] |= f[j - m]; //这边是或，因为避免之前可以，后面又不可以，覆盖掉了正确答案
                }
            }
        for(auto x : f) cout << x << " ";
        cout << endl;
        return f[n];
    }
};
```

另一种实现思路：(上面思路的效率更高)
```c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<int>dp(n + 1);
        dp[0] = 1;
        for (int i = 1; i <= s.size(); i++) 
            for (int j = 0; j < i; j++) 
                if (dp[j] && find(wordDict.begin(), wordDict.end(), s.substr(j , i - j)) != wordDict.end()) // s.substr(j + 1 - 1, i - ( j + 1) + 1)
                    dp[i] = 1;
        return dp[n];

    }
};
```
---


### 337. 打家劫舍 III
题意：一棵二叉树，树上的每个点都有对应的权值，每个点有两种状态（选中和不选中），问在不能同时选中有父子关系的点的情况下，能选中的点的最大权值和是多少。

`f[root]`表示选`root`点的最大值， `g[root]`表示不选`root`点的最大值
因此状态转移：
```c++
f[root] = root->val +  g[root->left] + g[root->right];
g[root] = max(g[root->left], f[root->left]) + max(g[root->right],f[root->right]);
```
```c++
class Solution {
public:
    unordered_map<TreeNode*, int>f, g; // f[root]表示选root点的最大值， g[root]表示不选root点的最大值
    int rob(TreeNode* root) {
        DFS(root);
        return max(g[root], f[root]);
    }
    void DFS(TreeNode* root) {
        if (!root) return;
        DFS(root->left);
        DFS(root->right);
        f[root] = root->val +  g[root->left] + g[root->right];
        g[root] = max(g[root->left], f[root->left]) + max(g[root->right], f[root->right]);
    }
};
```
时间复杂度$O(n)$:相当于对二叉树做了一次后序遍历
空间复杂度$O(n)$：栈空间 + 哈希表空间

**另一种写法：**  (考虑到每个节点在计算时，仅仅依赖于他的子节点)
```c++
class Solution {
public:
    vector<int> DFS(TreeNode* root) { // 0:不选，1:选
        if (!root) return {0, 0};
        vector<int> left = DFS(root->left);
        vector<int> right = DFS(root->right);
        int res1 = max(left[0], left[1]) + max(right[0], right[1]);
        int res2 = root->val + left[0] + right[0];
        return {res1, res2};
    }
    int rob(TreeNode* root) {
        auto ans = DFS(root);
        return max(ans[0], ans[1]);
    }
};
```
时间复杂度$O(n)$; 空间复杂度$O(n)$:虽说省去了哈希表的空间，但是还有栈空间

---



### 718. 最长重复子数组
题意：找到两个数组中最长的公共子数组，数组相比于序列要求连续

#### 思路1：DP
**时间复杂度：$O(nm)$**
```c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size();
        vector<vector<int>>dp(n + 1, vector<int>(m + 1, 0));
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j ++) {
                if (nums1[i - 1] == nums2[j - 1]) 
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                ans = max(ans, dp[i][j]);
            }
        }
        return ans;
    }
};
```

**优化到一维：**
```c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size();
        vector<int>dp(m + 1, 0);
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = m; j >= 1; j --) {
                if (nums1[i - 1] == nums2[j - 1]) 
                    dp[j] = dp[j - 1] + 1;
                else dp[j] = 0;
                ans = max(ans, dp[j]);
            }
        }
        return ans;
    }
};
```

#### 思路2：滑动窗口
我们可以枚举 `A` 和 `B` 所有的对齐方式。对齐的方式有两类：第一类为 `A` 不变，`B` 的首元素与 `A` 中的某个元素对齐；第二类为 `B` 不变，`A` 的首元素与 `B` 中的某个元素对齐。对于每一种对齐方式，我们计算它们相对位置相同的重复子数组即可。
**时间复杂度：$O((n+m)*(min(n,m)))$**

```c++
class Solution {
public:
    int cal(vector<int>& nums1, vector<int>& nums2, int p, int q) {
        int ans = 0, res = 0;
        int len1 = nums1.size() - p, len2 = nums2.size() - q;
        for (int k = 0; k < min(len1, len2); k++) {
            if (nums1[p + k] == nums2[q + k]) res++;
            else res = 0;
            ans = max(ans, res);
        }
        return ans;
    }
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int ans = 0;
        for (int i = 0; i < nums1.size(); i++) 
            ans = max(ans, cal(nums1, nums2, i, 0));
        for (int j = 0; j < nums2.size(); j++)
            ans = max(ans, cal(nums1, nums2, 0, j));
        return ans;
    }
};
```
---

### 53. 最大子数组和
`[5,4,-1,7,8], ans = 23`

#### 思路1：贪心
```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = 0, ans = -1e9;
        for (auto x : nums) {
            if (res < 0) res = 0;
            res += x;
            ans = max(ans, res);
        }
        return ans;
    }
};
```

#### 思路2：DP
`f[i]`:以`i`结尾的（包括`i`）的最大子数组的和
```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = nums[0];
        vector<int>f(nums.size());
        f[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            f[i] = max(f[i - 1] + nums[i], nums[i]);
            ans = max(ans, f[i]);
        }
        return ans;
        
    }
};
```
---

### 392. 判断子序列
题意：给定字符串 `s` 和 `t` ，判断 `s` 是否为 `t` 的子序列。
`s = "abc", t = "ahbgdc", ans = 1`

#### 思路1：简单
```c++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int k = 0;
        for (auto c : t) {
            if (c == s[k]) k++;
            if (k == s.size()) break;
        }
        return k == s.size() ? 1 : 0;
    }
};
```

进阶：如果有很多个`s`需要判断是否是`t`的子序列?
提前预处理出`t`的信息
```c++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int f[10010][26]; // f[i][c]表示从t的第i个字符往后字符c第一次出现的位置
        int n = s.size(), m = t.size(); 
        for (int j = 0; j < 26; j++) f[m][j] = m;
        for (int i = m - 1; i >= 0; i--) 
            for (int j = 0; j < 26; j++) 
                if (t[i] == 'a' + j) f[i][j] = i;
                else f[i][j] = f[i + 1][j]; 

        int pos = 0;
        for (int i = 0; i < n; i++) {
            pos = f[pos][s[i] - 'a'];
            if (pos == m) return 0;
            pos += 1;
        }
        return 1;
            
        
    }
};
```
---

### 115. 不同的子序列
题意：给定一个字符串`s `和一个字符串 `t` ，计算在 `s` 的子序列中 `t` 出现的个数。
`s = "babgbag", t = "bag", ans = 5`

#### 在纸上画画图，模拟下过程 

```c++
class Solution {
public:
    typedef unsigned long long ll;
    int numDistinct(string s, string t) {
        int n = t.size(), m = s.size();
        vector<vector<ll>>f(n + 1, vector<ll>(m + 1, 0));
        for(int j = 0; j <= m; j++) f[0][j] = 1;
        for (int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++){
                f[i][j] = f[i][j - 1];
                if (t[i - 1] == s[j - 1])  
                    f[i][j] += f[i - 1][j - 1];
            }
        }
        return f[n][m];
    }
};
```
---

### 647. 回文子串
给你一个字符串 `s` ，请你统计并返回这个字符串中 **回文子串** 的数目。
`s = "aaa", ans = 6`

#### 思路1：双指针
枚举每个回文字串中心点的位置，分奇偶

```c++
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += 1; //单个字符都是回文
            int l = i - 1, r = i + 1;
            while(l >= 0 && r < n && s[l--] == s[r++]) ans++;
        }
        for (int i = 0; i < n; i++) {
            int j = i + 1;
            if (s[i] != s[j]) continue;
            ans++;
            int l = i - 1, r = j + 1;
            while(l >= 0 && r < n && s[l--] == s[r++]) ans++;
        }
        return ans;
    }
};
```

#### 思路2：DP
`f[i][j]`表示`i~j`范围内的字符串是否是回文的
```c++
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size();
        vector<vector<int>>f(n + 1, vector<int>(n + 1, 0));
        int ans = 0;
        // 注意遍历顺序 判断f[i][j] 需要知道f[i + 1][j - 1]
        for (int i = n - 1; i >= 0; i--) { 
            for (int j = i; j < n; j++) {
                if(s[i] == s[j]) {
                    if (j - i <= 1) {
                        ans++;
                        f[i][j] = 1;
                    }
                    else if(f[i + 1][j - 1]) {
                        ans++;
                        f[i][j] = 1;
                    }
                }
            }
        }
        return ans;
    }
};
```
---

### 516. 最长回文子序列
给你一个字符串`s` ，找出其中最长的回文**子序列**，并返回该序列的长度。
注意是序列，不是字串
`s = "bbbab", ans = 4`
```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>>f(n, vector<int>(n, 0));
        for (int i = n - 1; i >= 0; i--) {
            f[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s[i] == s[j]) f[i][j] = f[i + 1][j - 1] + 2;
                else f[i][j] = max(f[i + 1][j], f[i][j - 1]);
            }
        }
        return f[0][n - 1];
    }
};
```
---

### 376. 摆动序列
如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。
`nums = [1,2,3,4,5,6,7,8,9], ans = 2`

#### 思路1：DP
**实现思路1**
**时间复杂度O($n^2$)**
```c++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        int ans = 1;
        vector<vector<int>>f(n, vector<int>(2, 1));//f[i][0]:以i结尾前面是升序，f[i][1]:以i结尾前面是降序
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) f[i][0] = max(f[i][0], f[j][1] + 1);
                else if (nums[i] < nums[j]) f[i][1] = max(f[i][1], f[j][0] + 1);
            }
            ans = max(ans, max(f[i][0], f[i][1]));
        }
        return ans;
    }
};
```

**实现思路2**
**时间复杂度O($n$)**
```c++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        vector<int>up(n, 1), down(n, 1);
        // up[i]表示前i个元素，以某个元素作为结尾的最长摆动序列，最后趋势是上升
        // down[i]表示前i个元素，以某个元素作为结尾的最长摆动序列，最后趋势是下降
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) {
                up[i] = max(down[i - 1] + 1, up[i - 1]);
                down[i] = down[i - 1];
            }
            else if (nums[i] < nums[i - 1]) {
                down[i] = max(down[i - 1], up[i - 1] + 1);
                up[i] = up[i - 1];
            }
            else {
                up[i] = up[i - 1];
                down[i] = down[i - 1];
            }
        }
        return max(up[n - 1], down[n - 1]);
    }
};
```

#### 思路2：贪心
求原序列峰和谷的数量就是答案。相当于找极值点
```c++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int ans = 1;
        int curdiff = 0, prediff = 0;
        for (int i = 1; i < nums.size(); i++) {
            curdiff = nums[i] - nums[i - 1];
            if ((curdiff > 0 && prediff <= 0)|| (curdiff < 0 && prediff >= 0)) {
                ans++;
                prediff = curdiff;
            }
        }
        return ans;
    }
};
```
---

### 45. 跳跃游戏 II
数组`nums`每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。
返回到达 `nums[n - 1]` 的最小跳跃次数。
`nums = [2,3,1,1,4], ans = 2`

#### 思路1：DP
```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        vector<int>f(n, 1e8);
        f[0] = 0;
        for (int i = 1; i < n; i++) 
            for (int j = 0; j < i; j++) 
                if (j + nums[j] >= i) f[i] = min(f[i], f[j] + 1);
        return f[n - 1];

    }
};
```

#### 贪心
```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int k = 0, end = 0, step = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
            k = max(k, i + nums[i]);
            if (i == end) {
                end = k;
                step++;
            }
        }
        return step;

    }
};
```
---

### 剑指 Offer 46. 把数字翻译成字符串
`0` 翻译成 `“a”` ，`1` 翻译成 `“b”`，……，`11` 翻译成 `“l”`，……，`25` 翻译成 `“z”`。一个数字可能有多个翻译。
问一共有多少种翻译

`12258, ans = 5`

#### DP
```c++
class Solution {
public:
    int f[15]; // f[i]:以s[i]为结尾的字符串的翻译个数
    bool check(string &s, int i) { // 判断s[i-1]和s[i]组成的数在不在25以内
        int a = s[i - 1]- '0', b = s[i] - '0';
        int num = a * 10 + b;
        if(num <= 25 && num >= 10) return 1; 
        return 0;
    }
    int translateNum(int num) {
        string s = " " + to_string(num);
        int n = s.size() - 1; // 不加前面空格的长度
        memset(f, 0, sizeof(f));
        f[0] = 1;
        for(int i = 1; i <= n; i++) {
            f[i] += f[i - 1]; // s[i]单独为一组。
            if(i >= 2 && check(s, i)) {
                f[i] += f[i - 2]; // s[i-1]s[i]为一组，上一个以s[i-2]结尾
            }
        }
        return f[n];
    }
};
```
---


### 剑指 Offer 62. 圆圈中最后剩下的数字
`0,1,···,n-1`这`n`个数字排成一个圆圈，从数字`0`开始，每次从这个圆圈里删除第`m`个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

#### 约瑟夫环问题

状态表示：`dp[i]`: i个人的情况下最后剩下的人所在的位置。
初始状态：`dp[1] = 0`
状态转移: `dp[i] = (dp[i - 1] + m) % i`

<img src="../Fig/约瑟夫环1.png">
<img src="../Fig/约瑟夫环2.png">

```c++
class Solution {
public:
    int lastRemaining(int n, int m) {
        int dp = 0;
        for(int i = 2; i <= n; i++) {
            dp = (dp + m) % i;
        }
        return dp;
    }
};
```
---

### 375. 猜数字大小 II
我从 `1` 到 `n` 之间选择一个数字。你来猜我选了哪个数字。
如果猜`x`,但是猜错了，需要支付`x`，问在保证可以猜对的情况下最少支付多少钱。

#### DP
```c++
class Solution {
public:
    int getMoneyAmount(int n) {
        vector<vector<int>>dp(n + 1, vector<int>(n + 1, 0));
        // dp[i][j]：猜区间i~j之间的数字，最少需要支付多少。  dp[i][i] = 0
        for(int len = 2; len <= n; len++) {
            for(int i = 1; i + len - 1 <= n; i++) {
                int j = i + len - 1;
                dp[i][j] = min(i + dp[i + 1][j], j + dp[i][j - 1]); // 边界
                for(int k = i + 1; k < j; k++) { // 以k作为分割点
                    dp[i][j] = min(dp[i][j], k + max(dp[i][k - 1], dp[k + 1][j]));
                }
            }
        }
        return dp[1][n];
    }
};
```
---