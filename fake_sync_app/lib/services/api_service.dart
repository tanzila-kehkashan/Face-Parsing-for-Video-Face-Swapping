import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import '../models/job.dart';
import 'dart:convert';
import 'dart:async';
class NSFWContentException implements Exception {
  final String message;
  NSFWContentException(this.message);

  @override
  String toString() => message;
}
class ApiService {
  late final Dio _dio;
  final String baseUrl = 'https://104edaa6d03b.ngrok-free.app'; // Your ngrok URL
  CancelToken? _currentUploadCancelToken;
  void cancelCurrentUpload() {
    if (_currentUploadCancelToken != null && !_currentUploadCancelToken!.isCancelled) {
      _currentUploadCancelToken!.cancel('Upload cancelled by user');
      _currentUploadCancelToken = null;
      print('Upload cancelled');
    }
  }
  ApiService() {
    // Create Dio with extended timeouts for video processing
    _dio = Dio(BaseOptions(
      connectTimeout: const Duration(minutes: 5),    // 5 minutes to connect
      receiveTimeout: const Duration(minutes: 45),   // 45 minutes to receive response
      sendTimeout: const Duration(minutes: 10),      // 10 minutes to send
      validateStatus: (status) {
        return status != null && status < 500;
      },
      headers: {
        'ngrok-skip-browser-warning': 'true', // Skip ngrok browser warning
      },
    ));


    // Add request interceptor for debugging
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) {
        print('REQUEST: ${options.method} ${options.uri}');
        print('HEADERS: ${options.headers}');
        handler.next(options);
      },
      onResponse: (response, handler) {
        print('RESPONSE: ${response.statusCode} ${response.requestOptions.uri}');
        if (response.data != null) {
          String responseStr = response.data.toString();
          String truncatedResponse = responseStr.length > 200
              ? responseStr.substring(0, 200) + '...'
              : responseStr;
          print('RESPONSE DATA: $truncatedResponse');
        }
        handler.next(response);
      },
      onError: (error, handler) {
        print('ERROR: ${error.message}');
        print('ERROR TYPE: ${error.type}');
        print('RESPONSE CODE: ${error.response?.statusCode}');
        print('RESPONSE DATA: ${error.response?.data}');
        handler.next(error);
      },
    ));
  }

  // Upload files and start processing with improved error handling
  Future<String> uploadFiles(File sourceImage, File targetVideo, [String? settings]) async {
    try {
      _currentUploadCancelToken = CancelToken();
      print('Starting file upload...');
      print('Source image size: ${await sourceImage.length()} bytes');
      print('Target video size: ${await targetVideo.length()} bytes');

      // Verify files exist and are readable
      if (!await sourceImage.exists()) {
        throw Exception('Source image file does not exist');
      }
      if (!await targetVideo.exists()) {
        throw Exception('Target video file does not exist');
      }

      // Validate file sizes
      final sourceSize = await sourceImage.length();
      final targetSize = await targetVideo.length();

      if (sourceSize == 0) {
        throw Exception('Source image file is empty');
      }
      if (targetSize == 0) {
        throw Exception('Target video file is empty');
      }

      // Check file size limits (500MB total)
      const maxSize = 500 * 1024 * 1024; // 500MB
      if (sourceSize + targetSize > maxSize) {
        throw Exception('Combined file size exceeds 500MB limit');
      }

      print('File validation passed. Creating form data...');

      Map<String, dynamic> formDataMap = {
        'source_image': await MultipartFile.fromFile(
          sourceImage.path,
          filename: 'source_image.jpg',
        ),
        'target_video': await MultipartFile.fromFile(
          targetVideo.path,
          filename: 'target_video.mp4',
        ),
      };

      // Only add settings if they exist (Best mode)
      if (settings != null) {
        formDataMap['settings'] = settings;
        print('DEBUG: Adding settings to form data: $settings');
      } else {
        print('DEBUG: No settings added - using server defaults');
      }

      FormData formData = FormData.fromMap(formDataMap);

      print('Sending upload request...');

      Response response = await _dio.post(
        '$baseUrl/upload/',
        data: formData,
        cancelToken: _currentUploadCancelToken, // Add this line
        options: Options(
          headers: {
            'Content-Type': 'multipart/form-data',
            'ngrok-skip-browser-warning': 'true',
          },
        ),
        onSendProgress: (sent, total) {
          if (total != -1) {
            double progress = (sent / total * 100);
            print('Upload progress: ${progress.toStringAsFixed(2)}%');
          }
        },
      );
      _currentUploadCancelToken = null;
      print('Upload response: ${response.statusCode}');
      print('Upload response data: ${response.data}');

      if (response.statusCode == 202) {
        String jobId = response.data['job_id'];
        print('Job created successfully: $jobId');
        return jobId;
      } else {
        throw Exception('Failed to upload files: ${response.statusMessage} - ${response.data}');
      }
    } on DioException catch (e) {
      print('DioException occurred: ${e.type}');
      print('DioException message: ${e.message}');
      print('DioException response: ${e.response?.data}');

      switch (e.type) {
        case DioExceptionType.connectionTimeout:
          throw Exception('Connection timeout - check your internet connection and server availability');
        case DioExceptionType.sendTimeout:
          throw Exception('Upload timeout - files may be too large or connection is slow');
        case DioExceptionType.receiveTimeout:
          throw Exception('Server response timeout - the server may be busy');
        case DioExceptionType.badResponse:
          final statusCode = e.response?.statusCode;
          final responseData = e.response?.data;

          // Handle specific NSFW-related errors
          if (statusCode == 400 && responseData != null) {
            String errorDetail = responseData['detail'] ?? 'Unknown error';

            // Check for NSFW-related error messages
            if (errorDetail.contains('Content blocked') ||
                errorDetail.contains('Inappropriate content') ||
                errorDetail.contains('rejected')) {
              throw NSFWContentException(errorDetail);
            }
          }

          throw Exception('Server error ($statusCode): ${responseData ?? 'Unknown error'}');
        case DioExceptionType.cancel:
          throw Exception('Upload was cancelled');
        case DioExceptionType.connectionError:
          throw Exception('Connection error - check your internet connection');
        default:
          throw Exception('Upload failed: ${e.message}');
      }
    } catch (e) {
      if (e is NSFWContentException) {
        rethrow; // Re-throw NSFW exceptions as-is
      }
      print('Unexpected error during upload: $e');
      throw Exception('Upload error: $e');
    }
  }

  // Check job status with retry logic and improved error handling
  Future<String> checkStatus(String jobId) async {
    int maxRetries = 3;
    int retryCount = 0;

    while (retryCount < maxRetries) {
      try {
        Response response = await _dio.get(
          '$baseUrl/jobs/$jobId/status/',
          options: Options(
            headers: {
              'ngrok-skip-browser-warning': 'true',
            },
            receiveTimeout: const Duration(seconds: 30), // Shorter timeout for status checks
          ),
        );

        print('Status check response: ${response.statusCode} - ${response.data}');

        if (response.statusCode == 200) {
          String status = response.data['status'];

          // Additional info logging
          if (response.data['message'] != null) {
            print('Job message: ${response.data['message']}');
          }
          if (response.data['file_size'] != null) {
            print('Output file size: ${response.data['file_size']} bytes');
          }

          return status;
        } else {
          throw Exception('Failed to check status: ${response.statusMessage}');
        }
      } on DioException catch (e) {
        retryCount++;
        print('Status check DioException (attempt $retryCount/$maxRetries): ${e.type} - ${e.message}');

        if (retryCount >= maxRetries) {
          throw Exception('Status check failed after $maxRetries attempts: ${e.message}');
        }

        // Wait before retry (exponential backoff)
        await Future.delayed(Duration(seconds: 2 * retryCount));
      } catch (e) {
        retryCount++;
        print('Status check error (attempt $retryCount/$maxRetries): $e');

        if (retryCount >= maxRetries) {
          throw Exception('Status check failed after $maxRetries attempts: $e');
        }

        // Wait before retry
        await Future.delayed(Duration(seconds: 2 * retryCount));
      }
    }

    throw Exception('Status check failed after maximum retries');
  }

  // Download result video with progress tracking and improved error handling
  Future<File> downloadResult(String jobId) async {
    try {
      final tempDir = await getTemporaryDirectory();
      final String savePath = '${tempDir.path}/swapface_result_$jobId.mp4';

      print('Downloading result to: $savePath');

      // First check if the file is ready
      Response statusResponse = await _dio.get(
        '$baseUrl/jobs/$jobId/status/',
        options: Options(
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      if (statusResponse.data['status'] != 'completed') {
        throw Exception('Job not completed yet: ${statusResponse.data['status']}');
      }

      print('Job confirmed as completed, starting download...');

      // Download file with progress tracking
      await _dio.download(
        '$baseUrl/jobs/$jobId/result',
        savePath,
        options: Options(
          responseType: ResponseType.bytes,
          receiveTimeout: const Duration(minutes: 10), // 10 minutes for download
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        ),
        onReceiveProgress: (received, total) {
          if (total != -1) {
            double progress = (received / total * 100);
            print('Download progress: ${progress.toStringAsFixed(2)}%');
          } else {
            print('Downloaded: ${(received / 1024 / 1024).toStringAsFixed(2)} MB');
          }
        },
      );

      final file = File(savePath);
      if (!file.existsSync()) {
        throw Exception('Downloaded file not found at expected location');
      }

      final fileSize = file.lengthSync();
      if (fileSize == 0) {
        throw Exception('Downloaded file is empty');
      }

      print('Download completed successfully: ${(fileSize / 1024 / 1024).toStringAsFixed(2)} MB');
      return file;
    } on DioException catch (e) {
      print('Download DioException: ${e.type} - ${e.message}');

      switch (e.type) {
        case DioExceptionType.receiveTimeout:
          throw Exception('Download timeout - the video file may be very large');
        case DioExceptionType.connectionTimeout:
          throw Exception('Connection timeout during download');
        case DioExceptionType.badResponse:
          throw Exception('Server error during download: ${e.response?.statusCode}');
        default:
          throw Exception('Download failed: ${e.message}');
      }
    } catch (e) {
      print('Download error: $e');
      throw Exception('Download error: $e');
    }
  }

  // Stream logs with better error handling
  Stream<String> streamLogs(String jobId) async* {
    print('Starting log stream for job: $jobId');

    final client = Dio(BaseOptions(
      baseUrl: baseUrl,
      responseType: ResponseType.stream,
      receiveTimeout: const Duration(minutes: 45), // Extended timeout for processing
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    ));

    try {
      final response = await client.get<ResponseBody>(
        '/jobs/$jobId/stream_logs',
        options: Options(
          headers: {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'ngrok-skip-browser-warning': 'true',
          },
        ),
      );

      print('Log stream response: ${response.statusCode}');

      if (response.statusCode == 200 && response.data?.stream != null) {
        String buffer = '';

        await for (var chunk in response.data!.stream!) {
          try {
            String data = utf8.decode(chunk);
            buffer += data;

            // Process complete lines
            while (buffer.contains('\n')) {
              int newlineIndex = buffer.indexOf('\n');
              String line = buffer.substring(0, newlineIndex).trim();
              buffer = buffer.substring(newlineIndex + 1);

              if (line.startsWith('data: ')) {
                String logData = line.substring(6);
                if (logData.isNotEmpty && logData != ' ') {
                  yield logData;
                }
              } else if (line.startsWith('event: end')) {
                print('Received end event from server');
                return;
              }
            }
          } catch (e) {
            print('Error processing stream chunk: $e');
            yield 'Error processing log data: $e';
          }
        }
      } else {
        throw Exception('Failed to stream logs: ${response.statusCode}');
      }
    } on DioException catch (e) {
      print('Stream logs DioException: ${e.type} - ${e.message}');

      switch (e.type) {
        case DioExceptionType.receiveTimeout:
          yield 'Log stream timeout - processing may still be ongoing';
        case DioExceptionType.connectionTimeout:
          yield 'Connection timeout - unable to connect to log stream';
        case DioExceptionType.connectionError:
          yield 'Connection error - check your internet connection';
        default:
          yield 'Error streaming logs: ${e.message}';
      }
    } catch (e) {
      print('Stream logs error: $e');
      yield 'Error streaming logs: $e';
    } finally {
      try {
        client.close();
      } catch (e) {
        print('Error closing stream client: $e');
      }
    }
  }

  // Get list of all jobs
  Future<List<Job>> getJobs() async {
    try {
      Response response = await _dio.get(
        '$baseUrl/jobs/',
        options: Options(
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      if (response.statusCode == 200) {
        List<dynamic> jobsData = response.data['jobs'];
        return jobsData.map((jobData) => Job.fromJson(jobData)).toList();
      } else {
        throw Exception('Failed to get jobs: ${response.statusMessage}');
      }
    } on DioException catch (e) {
      print('Get jobs DioException: ${e.type} - ${e.message}');
      throw Exception('Failed to get jobs: ${e.message}');
    } catch (e) {
      print('Get jobs error: $e');
      throw Exception('Get jobs error: $e');
    }
  }

  // Delete a job
  Future<bool> deleteJob(String jobId) async {
    try {
      Response response = await _dio.delete(
        '$baseUrl/jobs/$jobId',
        options: Options(
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      if (response.statusCode == 200) {
        return true;
      } else {
        throw Exception('Failed to delete job: ${response.statusMessage}');
      }
    } on DioException catch (e) {
      print('Delete job DioException: ${e.type} - ${e.message}');
      throw Exception('Failed to delete job: ${e.message}');
    } catch (e) {
      print('Delete job error: $e');
      throw Exception('Delete job error: $e');
    }
  }

  // Cancel a running job
  Future<Map<String, dynamic>> cancelJob(String jobId) async {
    try {
      Response response = await _dio.post(
        '$baseUrl/jobs/$jobId/cancel',
        options: Options(
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      print('Cancel job response: ${response.statusCode} - ${response.data}');

      if (response.statusCode == 200) {
        return {
          'success': true,
          'message': response.data['message'] ?? 'Job cancelled successfully',
          'status': response.data['status'] ?? 'cancelled'
        };
      } else {
        return {
          'success': false,
          'message': 'Failed to cancel job: ${response.statusMessage}'
        };
      }
    } on DioException catch (e) {
      print('Cancel job DioException: ${e.type} - ${e.message}');
      return {
        'success': false,
        'message': 'Cancel job failed: ${e.message}'
      };
    } catch (e) {
      print('Cancel job error: $e');
      return {
        'success': false,
        'message': 'Cancel job error: $e'
      };
    }
  }

  // Test connection to server
  Future<bool> testConnection() async {
    try {
      Response response = await _dio.get(
        '$baseUrl/health',
        options: Options(
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
          receiveTimeout: const Duration(seconds: 10),
        ),
      );

      print('Connection test response: ${response.data}');
      return response.statusCode == 200;
    } on DioException catch (e) {
      print('Connection test DioException: ${e.type} - ${e.message}');
      return false;
    } catch (e) {
      print('Connection test failed: $e');
      return false;
    }
  }
  // Add method to check server NSFW detection status
  Future<Map<String, dynamic>> getServerCapabilities() async {
    try {
      Response response = await _dio.get(
        '$baseUrl/health',
        options: Options(
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
          receiveTimeout: const Duration(seconds: 10),
        ),
      );

      if (response.statusCode == 200) {
        final data = response.data;
        return {
          'nsfw_detection_enabled': data['server_config']?['nsfw_detection'] ?? false,
          'nsfw_threshold': data['server_config']?['nsfw_threshold'],
          'content_safety': data['features']?['content_safety'] ?? 'Unknown',
          'models': data['models'] ?? {},
        };
      } else {
        throw Exception('Failed to get server capabilities: ${response.statusCode}');
      }
    } on DioException catch (e) {
      print('Get server capabilities DioException: ${e.type} - ${e.message}');
      return {
        'nsfw_detection_enabled': false,
        'error': 'Failed to check server capabilities: ${e.message}'
      };
    } catch (e) {
      print('Get server capabilities error: $e');
      return {
        'nsfw_detection_enabled': false,
        'error': 'Failed to check server capabilities: $e'
      };
    }
  }
}