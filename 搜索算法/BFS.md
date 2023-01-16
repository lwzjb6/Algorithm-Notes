<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> BFS </font> </center>

**适用场景：找最短路，最少变换次数等**

### 经典问题：迷宫找最短路
```
问从左上角走到右下角最少需要几步？
输入
5 5     迷宫大小
0 1 0 0 0   0表示可走，1表示障碍物
0 1 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 1 0
输出
8
```

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 110;
int n, m;
int a[N][N], vis[N][N];
struct node{
    int x, y;
    int step;
};
int dir[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
bool check(int x, int y) {
    if(x < 0 || x >= n || y < 0 || y >= m || vis[x][y] || a[x][y]) return 0;
    return 1;
}
int main()
{
    
    cin >> n >> m;
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < m; j++)
            cin >> a[i][j];
    queue<node>q;
    q.push({0, 0, 0});
    while(q.size()) {
        auto e = q.front();
        q.pop();
        
        if(e.x == n - 1 && e.y == m - 1){
            cout << e.step << endl;
            break;
        }
        
        for(int i = 0; i < 4; i ++) {
            int nx = e.x + dir[i][0];
            int ny = e.y + dir[i][1];
            if(check(nx, ny)) {
                q.push({nx, ny, e.step + 1});
                vis[nx][ny] = 1;
            }
        }
    }
    
    return 0;
}
```
---

### 面试题13. 机器人的运动范围
机器人从`（0,0）`开始走，每次只能向4个方向移动一格，不能进入数位之和大于`k`的格子，例如：当`k=18`时，机器人不能进入方格 `[35, 38]`, 因为`3 + 5 + 3 + 8 > 18`。问最多进入遍历多少个格子。

易错点：容易想当然认为边界是一个斜线，因为如果当前在`(x, y)`, 并且`（x, y）`的数位之和等于k, 因此下一行的边界为`(x + 1, y - 1)`。但实际上`(9, 9)`的数位之和为`18`，`(9, 10)`的数位之和为`10`

#### BFS
```c++
class Solution {
public:
    typedef pair<int, int>pii;
    bool vis[110][110];
    int digit_sum(int x, int y) {
        int res = 0;
        while(x) {
            res += x % 10;
            x /= 10;
        }
        while(y) {
            res += y % 10;
            y /= 10;
        }
        return res;
    }
    int movingCount(int m, int n, int k) {
        queue<pii>q;
        q.push({0, 0});
        vis[0][0] = 1;
        int ans = 0;
        while(q.size()) {
            auto [x, y] = q.front();
            q.pop();
            ans ++;
            
            if(x + 1 < m && !vis[x + 1][y] && digit_sum(x + 1, y) <= k) {
                q.push({x + 1, y});
                vis[x + 1][y] = 1;
            }
            if(y + 1 < n && !vis[x][y + 1] && digit_sum(x, y + 1) <= k) {
                q.push({x, y + 1});
                vis[x][y + 1] = 1;
            }
        }  
        return ans;
    }
};
```
