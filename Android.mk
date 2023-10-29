LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := recognition
LOCAL_SRC_FILES := recognition.c main.c
include $(BUILD_SHARED_LIBRARY)
