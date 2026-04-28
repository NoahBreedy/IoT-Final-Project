#include <curl/curl.h>
#include <string>
#include <cstring>
#include <thread>
#include <vector>
#include <iostream>

using namespace std;

constexpr const char* const ips[] = {
    "PLACEHOLDER",
}; 


string get_capture_url(int robot_num) {
    return (
        "http://" + 
        string(ips[robot_num]) +
        "/capture"
    );
}


size_t write_callback(void* ptr, size_t size, size_t nmeb, vector<uint8_t>* buffer) {
    size_t total = size*nmeb;
    buffer->insert(buffer->end(), (uint8_t*)ptr, (uint8_t*)ptr + total);
    return total;
}

void get_capture(int robot, vector<uint8_t>& data) {
    CURL* curl = curl_easy_init();

    data.clear();

    string url = get_capture_url(robot);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
}

void print_img(void* data, size_t size, size_t row_size=64) {
    int i=0;
    row_size = max(row_size, size_t(16));

    uint8_t* bytes = (uint8_t*)data;

    for (;;) {
        if (i >= size) return;
        printf("%X", bytes[i]);
    
        ++i;
        if (i % 60 == 0) cout << endl;
    }
}


int main() {
    vector<uint8_t> data;
    data.reserve(3*360*360);
    for (;;) {
        get_capture(0, data);

        print_img((void*)data.data(), data.size());

        cout << "~" << endl;
        this_thread::sleep_for(chrono::milliseconds(300));
    }
}