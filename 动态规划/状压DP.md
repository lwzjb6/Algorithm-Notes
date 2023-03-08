<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 状压DP </font> </center>




### 982. 按位与为零的三元组
在数组中找到`3`个下标，可以相同，求使得`nums[i] & nums[j] & nums[k] == 0` 出现的次数。

```c++
数据范围
n = nums.size <= 1000
m = nums[i] <= 2 ^ 16
```

#### 哈希 + 暴力枚举
首先两重循环枚举出`nums[i] & nums[j]`对应数值出现的次数，哈希表记录，其可能的情况最多为`m`.

```c++
class Solution {
public:
    int countTriplets(vector<int>& nums) {
        unordered_map<int, int>hx;
        for(auto x : nums) 
            for(auto y : nums) 
                hx[x & y]++;     
        int ans = 0;
        for(auto x : nums) {
           for(auto [k, v] : hx) {
               if((x & k) == 0) ans += v;
           } 
        }
        return ans;
    }
};
```
**时间复杂度$O(nm)$**

#### 状态压缩
把二进制数的表示看成集合的表示，`1`表示对应为在集合中，`0`表示对应位不在集合中。
`nums[i] & nums[j] == 0` 表示 `i`和`j`的集合表示没有交集

```c++
class Solution {
public:
    int countTriplets(vector<int>& nums) {
        unordered_map<int, int>hx;
        for(auto x : nums) 
            for(auto y : nums) 
                hx[x & y]++;
        
        int M = (1 << 16);
        int ans = 0;
        for(auto x : nums) {
            int other = (M - 1) ^ x;  // x的补集
            // 子集枚举
            int j = other; // j 就表示 不会与x出现交集的集合表示
            do {
                ans += hx[j];
                j = (j - 1) & other;
            }while(j != other);
        }
        return ans;
    }
};
```
---



### 6364. 无平方子集计数 [good]

返回数组 `nums` 中无平方子集的数目（非空）。

无平方子集定义：如果数组 `nums` 的子集中的元素乘积是一个 无平方因子数 ，则认为该子集是一个无平方子集。

数据范围： `nums.size() <= 1000. nums[i] <= 30`

思路分析：对每个数进行质因数分解，选出的子集中的元素只要不含相同的质因子就是一个合法的无平方子集。

难点：如何记录一个数的质因数分解状态：**状态压缩。**
因为`30`以内的质数为`{2,3,5,7,11,13,17,19,23,29}`
如果某个数存在某个质因子，则其对应的二进制位为`1`
例如：`10 = 2 * 5  -> mask = 101 = 5`


#### 背包
首先对所有的数进行状压表示，然后将问题转换为，对于某个体积`j`,考虑用物品装满体积`j`的方案数`f[j]`。然后`sum(f[j])`就可找到答案。

状态表示：`f[i][j]`: 表示考虑前`i`数字，体积为`j`的方案数。
状态转移：`f[i][j] = f[i - 1][j] + f[i - 1][j ^ mask]`
`mask`表示第`i`个数的状压表示，如果要加后面这一项，前提是`mask`是`j`的子集, 异或得到前`i-1`个数占据的体积。

倒序枚举体积压缩到一维。

```c++
class Solution {
public:
    using ll = long long;
    const int mod = 1e9 + 7;
    int squareFreeSubsets(vector<int>& nums) {
        vector<int>pp = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        vector<int>hx(31, 0); // 得到每个数的状态表示
        for(int x = 2; x <= 30; x++) { // x= 1的mask=0
            // 得到x的状压表示
            for(int j = 0; j < pp.size(); j++) {
                if(x % pp[j] == 0) { // pp[j]是x的一个质因数
                    hx[x] |= (1 << j);
                    if(x % (pp[j] * pp[j]) == 0) { // x本身并不是无平方数
                        hx[x] = -1;
                        break;
                    }
                }
            }
        }
        // 01背包求方案数
        int m = (1 << pp.size());
        vector<int>f(m);
        f[0] = 1; // 背包体积为0的情况，加数字1不会增大背包的体积，因此也在f[0]中
        for(auto x : nums) {
            int mask = hx[x];
            if(mask == -1) continue; // 当前物品不能选
            for(int j = m - 1; j >= mask; j--) {
                if((j | mask) == j) { // mask是j的子集
                    f[j] = (f[j] + f[j ^ mask]) % mod; // 不选 + 选
                }
            } 
        }
        ll ans = accumulate(f.begin(), f.end(), 0L) % mod - 1;  // 减去空集
        return ans;
    }
};
```
**时间复杂度$O(2^{10} N)$**

#### 状压DP
同样对所有的数进行状压表示，一共`10`个质数，相当于一共$2^{10}$种可能的状态。
状态表示：`f[i]`表示状态为`i`的方案数
状态转移：假设当前数的状压表示为：`mask`, 其出现的次数为`count(mask)`, 其补码为`other`
`oher`的所有子集表示为`{j}`
`f[j | mask] += f[j] * count(mask)`
即用当前的数更新其能更新的状态。


```c++
class Solution {
public:
    using ll = long long;
    const int mod = 1e9 + 7;
    int squareFreeSubsets(vector<int>& nums) {
        vector<int>pp = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        vector<int>hx(31, 0); // 得到每个数的状态压缩表示
        for(int x = 2; x <= 30; x++) { // x= 1的mask=0
            // 得到x的状压表示
            for(int j = 0; j < pp.size(); j++) {
                if(x % pp[j] == 0) { // pp[j]是x的一个质因数
                    hx[x] |= (1 << j);
                    if(x % (pp[j] * pp[j]) == 0) { // x本身并不是无平方数
                        hx[x] = -1;
                        break;
                    }
                }
            }
        }
        vector<int>cnt(31, 0); // 每个数字出现的次数
        int num1 = 1; //  1 可以构成的子集的个数， 2^{1出现的次数}
        for(auto x : nums) {
            if(x == 1) num1 = num1 * 2 % mod;
            else cnt[x] ++;
        }
        
        // 状压DP
        int M = (1 << pp.size()); // 一共2^10种状态
        vector<int>f(M, 0);
        f[0] = 1;
        for(int x = 2; x <= 30; x++) {
            int mask = hx[x], c = cnt[x];
            if(c == 0) continue;
            if(mask <= 0) continue; // 不能选的数以及1先不考虑
            int other = (M - 1) ^ mask; // mask的补码
            // 枚举other的子集，可以举例101理解
            int j = other;
            do {
                f[j | mask] = (ll) (f[j | mask] + (ll) f[j] * c % mod) % mod;
                j = (j - 1) & other;
            }while(j != other);
        }
        ll ans = accumulate(f.begin(), f.end(), 0L) % mod;
        ans = (ans * num1) % mod; // 每个集合都可以和由1组成的子集拼接。
        return ans - 1; // 减去空集的个数
    }
};
```

**时间复杂度$O(2^{10} \times 30)$**