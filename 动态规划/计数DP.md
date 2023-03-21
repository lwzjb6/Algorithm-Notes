<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 计数DP </font> </center>

### acwing 900. 整数划分
将一个整数`n`划分为多个正整数之和，问有多少种方法？
`3 = 2 + 1 = 1 + 1 + 1 = 3`

转换为完全背包求方案数
```c++
#include<bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1e9 + 7;
int main(){
    int n;
    cin >> n;
    vector<int>f(n + 1); 
    f[0] = 1;
    for(int i = 1; i <= n; i++) { // 枚举物品
        for(int j = i; j <= n; j++)  // 完全背包，正序枚举体积
            f[j] = (f[j - i] + f[j]) % mod;
    }
    cout << f[n] << endl;
    return 0;
}
```

#### DP
状态表示：`f[i][j]`表示和为`i`, 划分成`j`个数的方案数
状态转移：将所有的方案分为两个集合。
（1）集合`1`：最小的数为`1`： `f[i][j] = f[i - 1][j - 1]` [可以由之前的方案 + 一个数字1构成]
（2）集合`2`：最小的数不为`1`：`f[i][j] = f[i - j][j]`[所有的数都减去1]
`f[i][j] = f[i - 1][j - 1] + f[i - j][j]`

答案：`sum(f[n][k] for k in range(1, n + 1))`

```c++
#include<bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1e9 + 7;
int f[1010][1010];
int main(){
    int n;
    cin >> n;
    
    f[0][0] = 1;
    for(int i = 1; i <= n; i++) 
        for (int j = 1; j <= i; j++) // 和为i的数最多由i个数构成
            f[i][j] = (f[i - 1][j - 1] + f[i - j][j]) % mod;
    
    ll ans = 0;
    for(int i = 1; i <= n; i++) ans = (ans + f[n][i]) % mod;
    cout << ans << endl;
    return 0;
}
```