import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../../widgets/option_tile.dart';
import '../auth/login_page.dart';
import 'help_center_page.dart';
import 'linked_accounts_page.dart';
import 'personal_information_page.dart';
import 'privacy_security_page.dart';
import 'terms_page.dart';
import 'premium_page.dart';

class AccountPage extends StatelessWidget {
  const AccountPage({super.key});

  static const route = '/account';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            Row(
              children: [
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.arrow_back_ios_new),
                ),
                const Spacer(),
                const Text(
                  'Account',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                IconButton(
                  onPressed: () {},
                  icon: const Icon(Icons.settings_outlined),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
              ),
              child: Row(
                children: [
                  const CircleAvatar(
                    radius: 36,
                    backgroundColor: AppColors.primaryBlue,
                    child: Text(
                      'OW',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          'Olivia Wilson',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          'olivia.wilson@email.com',
                          style: TextStyle(color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: () {},
                    icon: const Icon(Icons.edit_outlined),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Settings',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
            OptionTile(
              icon: Icons.person_outline,
              title: 'Personal Information',
              subtitle: 'Name, email, and password',
              onTap: () =>
                  Navigator.pushNamed(context, PersonalInformationPage.route),
            ),
            OptionTile(
              icon: Icons.link,
              title: 'Linked Accounts',
              subtitle: 'Manage connected services',
              onTap: () =>
                  Navigator.pushNamed(context, LinkedAccountsPage.route),
            ),
            OptionTile(
              icon: Icons.security,
              title: 'Privacy & Security',
              subtitle: 'Data and account protection',
              onTap: () =>
                  Navigator.pushNamed(context, PrivacySecurityPage.route),
            ),
            const SizedBox(height: 24),
            const Text(
              'Support',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
            OptionTile(
              icon: Icons.help_outline,
              title: 'Help & Support',
              onTap: () => Navigator.pushNamed(context, HelpCenterPage.route),
            ),
            OptionTile(
              icon: Icons.description_outlined,
              title: 'Terms of Service',
              onTap: () => Navigator.pushNamed(context, TermsPage.route),
            ),
            OptionTile(
              icon: Icons.workspace_premium_outlined,
              title: 'Manage Subscription',
              onTap: () => Navigator.pushNamed(context, PremiumPage.route),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.pushNamedAndRemoveUntil(
                  context,
                  LoginPage.route,
                  (route) => false,
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: Colors.redAccent,
                minimumSize: const Size(double.infinity, 56),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                side: const BorderSide(color: Colors.redAccent),
              ),
              icon: const Icon(Icons.logout),
              label: const Text('Log Out'),
            ),
          ],
        ),
      ),
    );
  }
}
