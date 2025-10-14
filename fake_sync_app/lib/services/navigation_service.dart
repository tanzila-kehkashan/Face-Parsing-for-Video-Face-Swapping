import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import '../screens/metadata_extractor.dart';
import '../screens/title_screen.dart';
import '../screens/home_screen.dart';
import '../screens/result_screen.dart';
import '../screens/sign_up_screen.dart';
import '../screens/login_screen.dart';
import '../screens/email_verification_screen.dart';
import '../screens/terms_and_conditions_screen.dart';
import '../screens/user_profile_screen.dart';
import '../screens/user_setting_screen.dart';
import '../screens/error.dart';
import '../screens/help.dart';

class NavigationService {
  static final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

  static BuildContext? get context => navigatorKey.currentContext;

  // Navigation methods with custom transitions
  static Future<T?> pushNamed<T extends Object?>(
      String routeName, {
        Object? arguments,
        bool useCustomTransition = true,
      }) {
    if (useCustomTransition) {
      return Navigator.of(navigatorKey.currentContext!).push(
        _createRoute(routeName, arguments),
      );
    } else {
      return Navigator.of(navigatorKey.currentContext!).pushNamed(
        routeName,
        arguments: arguments,
      );
    }
  }

  // UPDATED: Simplified to only require email
  static void goToEmailVerification(String email) {
    pushNamed(
      '/email_verification',
      arguments: {
        'email': email,
      },
    );
  }

  static Future<T?> pushReplacementNamed<T extends Object?, TO extends Object?>(
      String routeName, {
        Object? arguments,
        TO? result,
        bool useCustomTransition = true,
      }) {
    if (useCustomTransition) {
      return Navigator.of(navigatorKey.currentContext!).pushReplacement(
        _createRoute(routeName, arguments),
        result: result,
      );
    } else {
      return Navigator.of(navigatorKey.currentContext!).pushReplacementNamed(
        routeName,
        arguments: arguments,
        result: result,
      );
    }
  }

  // CHANGED: Renamed to pushNamedAndRemoveAll for clarity
  static Future<T?> pushNamedAndRemoveAll<T extends Object?>(
      String routeName, {
        Object? arguments,
      }) {
    return Navigator.of(navigatorKey.currentContext!).pushAndRemoveUntil(
      _createRoute(routeName, arguments),
          (route) => false,
    );
  }

  static void pop<T extends Object?>([T? result]) {
    Navigator.of(navigatorKey.currentContext!).pop(result);
  }

  static bool canPop() {
    return Navigator.of(navigatorKey.currentContext!).canPop();
  }

  static void popUntil(String routeName) {
    Navigator.of(navigatorKey.currentContext!).popUntil(
          (route) => route.settings.name == routeName,
    );
  }

  // Custom route creation with transitions
  static Route<T> _createRoute<T>(String routeName, Object? arguments) {
    return PageRouteBuilder<T>(
      pageBuilder: (context, animation, secondaryAnimation) {
        return _getPageForRoute(routeName, arguments);
      },
      transitionsBuilder: (context, animation, secondaryAnimation, child) {
        return _buildTransition(animation, child, routeName);
      },
      settings: RouteSettings(name: routeName, arguments: arguments),
      transitionDuration: const Duration(milliseconds: 300),
    );
  }

  // FIXED: Updated to use uniqueId for ResultScreen
  static Widget _getPageForRoute(String routeName, Object? arguments) {
    switch (routeName) {
      case '/':
        return const TitleScreen();
      case '/signup':
        return const SignUpScreen();
      case '/login':
        return const LoginScreen();
      case '/email_verification':
        final args = arguments as Map<String, dynamic>?;
        return EmailVerificationScreen(
          email: args?['email'] ?? '',
        );
      case '/terms':
        return const TermsAndConditionsScreen();
      case '/profile':
        return const UserProfileScreen();
      case '/settings':
        return const userSettingScreen();
      case '/meta':
        return const VideoMetadataScreen();
      case '/home':
        return const HomeScreen();
      case '/error':
        return const ErrorSampleScreen();
      case '/help':
        return const HelpSupportScreen();
      case '/result':
        return ResultScreen(uniqueId: arguments as String? ?? '');
      default:
        return const TitleScreen();
    }
  }

  // Build custom transitions based on route
  static Widget _buildTransition(Animation<double> animation, Widget child, String routeName) {
    switch (routeName) {
      case '/':
        return FadeTransition(opacity: animation, child: child);
      case '/home':
        return SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(1.0, 0.0),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: animation,
            curve: Curves.easeInOut,
          )),
          child: child,
        );
      case '/result':
        return ScaleTransition(
          scale: Tween<double>(
            begin: 0.8,
            end: 1.0,
          ).animate(CurvedAnimation(
            parent: animation,
            curve: Curves.easeOutBack,
          )),
          child: FadeTransition(
            opacity: animation,
            child: child,
          ),
        );
      case '/email_verification':
        return SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0.0, 0.5),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: animation,
            curve: Curves.easeOutQuart,
          )),
          child: FadeTransition(
            opacity: animation,
            child: child,
          ),
        );
      default:
        return SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0.0, 1.0),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: animation,
            curve: Curves.easeInOut,
          )),
          child: child,
        );
    }
  }

  // Simplified utility methods for core navigation
  static void goToHome() {
    pushNamedAndRemoveAll('/home');
  }
  static void goTohelp() {
    pushNamedAndRemoveAll('/help');
  }
  static void goToHom() {
    pushNamedAndRemoveAll('/error');
  }

  static void meta() {
    pushNamedAndRemoveAll('/meta');
  }

  // FIXED: Updated to use uniqueId
  static void goToResult(String uniqueId) {
    pushReplacementNamed('/result', arguments: uniqueId);
  }

  static void resetToTitle() {
    pushNamedAndRemoveAll('/');
  }

  static void goToSignup() {
    pushNamedAndRemoveAll('/signup');
  }

  static void goToLogin() {
    pushNamedAndRemoveAll('/login');
  }

  static void goToTerms() {
    pushNamedAndRemoveAll('/terms');
  }

  // NEW: Added method for logout navigation
  static void goToLoginAfterLogout() {
    FirebaseAuth.instance.signOut();
    pushNamedAndRemoveAll('/login');
  }

  // Show snackbars with custom styling
  static void showSnackBar({
    required String message,
    Color? backgroundColor,
    Color? textColor,
    Duration? duration,
    SnackBarAction? action,
  }) {
    if (navigatorKey.currentContext != null) {
      ScaffoldMessenger.of(navigatorKey.currentContext!).showSnackBar(
        SnackBar(
          content: Text(
            message,
            style: TextStyle(
              color: textColor ?? Colors.white,
              fontSize: 16,
            ),
          ),
          backgroundColor: backgroundColor ?? const Color(0xFF2D2D2D),
          duration: duration ?? const Duration(seconds: 3),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          margin: const EdgeInsets.all(16),
          action: action,
        ),
      );
    }
  }

  static void showSuccessSnackBar(String message) {
    showSnackBar(
      message: message,
      backgroundColor: Colors.green.shade600,
      textColor: Colors.white,
    );
  }

  static void showErrorSnackBar(String message) {
    showSnackBar(
      message: message,
      backgroundColor: Colors.red.shade600,
      textColor: Colors.white,
    );
  }

  static void showInfoSnackBar(String message) {
    showSnackBar(
      message: message,
      backgroundColor: Colors.purple.shade600,
      textColor: Colors.white,
    );
  }

  // Confirmation dialog
  static Future<bool> showConfirmationDialog({
    required String title,
    required String message,
    String confirmText = 'Confirm',
    String cancelText = 'Cancel',
    Color? confirmColor,
  }) async {
    final result = await showDialog<bool>(
      context: navigatorKey.currentContext!,
      barrierDismissible: true,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2D2D2D),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        title: Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        content: Text(
          message,
          style: const TextStyle(
            color: Colors.grey,
            fontSize: 16,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => pop(false),
            child: Text(
              cancelText,
              style: const TextStyle(color: Colors.grey),
            ),
          ),
          TextButton(
            onPressed: () => pop(true),
            child: Text(
              confirmText,
              style: TextStyle(
                color: confirmColor ?? Colors.purple,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
    return result ?? false;
  }
}