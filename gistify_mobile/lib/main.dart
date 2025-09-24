import 'package:flutter/material.dart';

import 'ui/pages/auth/login_page.dart';
import 'ui/pages/auth/sign_up_page.dart';
import 'ui/pages/dashboard/main_shell.dart';
import 'ui/pages/settings/account_page.dart';
import 'ui/pages/settings/custom_models_page.dart';
import 'ui/pages/settings/data_storage_page.dart';
import 'ui/pages/settings/help_center_page.dart';
import 'ui/pages/settings/linked_accounts_page.dart';
import 'ui/pages/settings/personal_information_page.dart';
import 'ui/pages/settings/premium_page.dart';
import 'ui/pages/settings/privacy_policy_page.dart';
import 'ui/pages/settings/privacy_security_page.dart';
import 'ui/pages/settings/terms_page.dart';
import 'ui/pages/settings/theme_page.dart';
import 'ui/pages/settings/notifications_page.dart';
import 'ui/theme/app_theme.dart';

void main() {
  runApp(const GistifyApp());
}

class GistifyApp extends StatelessWidget {
  const GistifyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gistify',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      initialRoute: LoginPage.route,
      routes: {
        LoginPage.route: (_) => const LoginPage(),
        SignUpPage.route: (_) => const SignUpPage(),
        MainShell.route: (_) => const MainShell(),
        PremiumPage.route: (_) => const PremiumPage(),
        DataStoragePage.route: (_) => const DataStoragePage(),
        CustomModelsPage.route: (_) => const CustomModelsPage(),
        AccountPage.route: (_) => const AccountPage(),
        HelpCenterPage.route: (_) => const HelpCenterPage(),
        TermsPage.route: (_) => const TermsPage(),
        PrivacyPolicyPage.route: (_) => const PrivacyPolicyPage(),
        ThemePage.route: (_) => const ThemePage(),
        NotificationsPage.route: (_) => const NotificationsPage(),
        PersonalInformationPage.route: (_) => const PersonalInformationPage(),
        LinkedAccountsPage.route: (_) => const LinkedAccountsPage(),
        PrivacySecurityPage.route: (_) => const PrivacySecurityPage(),
      },
    );
  }
}
