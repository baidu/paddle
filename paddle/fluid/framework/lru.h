#pragma once
#include <vector>
#include <list>
#include <unordered_map>
#include <deque>
#include <queue>
#include <iostream>

namespace paddle {
namespace framework {

template <class K, class V>
class LRUCache {
public:
   LRUCache(int capacity) {
       this->capacity = capacity;
   }

   int Size() {
     std::unique_lock<std::mutex> lock(mutex_);
     return q.size();
   }

   V Get(K key) {
       std::unique_lock<std::mutex> lock(mutex_);
       return GetUnlock(key);
/*
       auto itr = m.find(key);
       if(itr == m.end()) {
           return -1;
       }
       auto pair = *(itr->second);
       q.erase(itr->second);
       m.erase(itr);
       q.push_front(pair);
       m[key] = q.begin();
       return pair.second;
*/
   }

   V GetUnlock(K key) {
       //std::unique_lock<std::mutex> lock(mutex_);
       auto itr = m.find(key);
       if(itr == m.end()) {
           return V();
       }
       auto pair = *(itr->second);
       q.erase(itr->second);
       m.erase(itr);
       q.push_front(pair);
       m[key] = q.begin();
       return pair.second;
   }
    
   void Put(K key, V value) {
       //std::cout << "put before lock" << std::endl;
       std::unique_lock<std::mutex> lock(mutex_);
       //std::cout << "put after lock" << std::endl;
       auto itr = m.find(key);
       if(itr == m.end()) {
           if(m.size() == capacity) {
               auto pair = q.back();
               q.pop_back();
               m.erase(m.find(pair.first));
           }
           q.push_front(std::make_pair(key, value));
           m[key] = q.begin();
       } else {
           auto pair = *(itr->second);
           pair.second = value;
           q.erase(itr->second);
           q.push_front(pair);
           m[key] = q.begin();
       }
       rand_queue.push(key);
   }

   V GetRandom() {
     std::unique_lock<std::mutex> lock(mutex_);
     int size = rand_queue.size();
     for(int i = 0; i < size; ++i) {
     //while(!rand_queue.empty()) {
       K key = rand_queue.front();
       rand_queue.pop();
       if (m.find(key) == m.end()) {
         continue;
       }
       rand_queue.push(key);
       return GetUnlock(key);
     }
     return V();
   }

   // <key, value>
   std::list<std::pair<K, V>> q;
   //using std::list<std::pair<K, V>>::iterator;
   //typedef std::list<std::pair<K, V>>::iterator iter;
   using iter = typename std::list<std::pair<K, V>>::iterator;
   std::unordered_map<K, iter> m;
   //std::unordered_map<K, std::list<std::pair<K, V>>::iterator> m;
   int capacity;
   std::queue<K> rand_queue;

  //std::deque<T> data_;
  //int capacity;
  std::mutex mutex_;

};

}
}
