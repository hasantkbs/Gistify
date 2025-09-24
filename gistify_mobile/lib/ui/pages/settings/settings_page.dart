import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../../widgets/option_tile.dart';
import 'account_page.dart';
import 'custom_models_page.dart';
import 'data_storage_page.dart';
import 'help_center_page.dart';
import 'premium_page.dart';
import 'privacy_policy_page.dart';
import 'terms_page.dart';
import 'theme_page.dart';
import 'notifications_page.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
          children: [
            Row(
              children: [
                IconButton(
                  onPressed: () => Navigator.maybePop(context),
                  icon: const Icon(Icons.arrow_back_ios_new),
                ),
                const Spacer(),
                const Text(
                  'Settings',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.3,
                  ),
                ),
                const Spacer(),
                const SizedBox(width: 48),
              ],
            ),
            const SizedBox(height: 32),
            const _SectionHeader('Account'),
            OptionTile(
              icon: Icons.person_outline,
              title: 'Account',
              subtitle: 'Manage your profile & subscription',
              onTap: () => Navigator.pushNamed(context, AccountPage.route),
            ),
            OptionTile(
              icon: Icons.workspace_premium_outlined,
              title: 'Premium',
              subtitle: 'Upgrade to unlock more features',
              onTap: () => Navigator.pushNamed(context, PremiumPage.route),
            ),
            const SizedBox(height: 12),
            const _SectionHeader('Preferences'),
            OptionTile(
              icon: Icons.dark_mode_outlined,
              title: 'Theme',
              subtitle: 'Light / Dark Mode',
              onTap: () => Navigator.pushNamed(context, ThemePage.route),
            ),
            OptionTile(
              icon: Icons.notifications_none,
              title: 'Notifications',
              subtitle: 'Manage your notifications',
              onTap: () =>
                  Navigator.pushNamed(context, NotificationsPage.route),
            ),
            const SizedBox(height: 12),
            const _SectionHeader('App Settings'),
            OptionTile(
              icon: Icons.storage_outlined,
              title: 'Data',
              subtitle: 'Manage your data',
              onTap: () => Navigator.pushNamed(context, DataStoragePage.route),
            ),
            OptionTile(
              icon: Icons.sd_storage_outlined,
              title: 'Storage',
              subtitle: 'Manage your storage',
              onTap: () => Navigator.pushNamed(context, DataStoragePage.route),
            ),
            OptionTile(
              icon: Icons.hub_outlined,
              title: 'Custom Models',
              subtitle: 'Manage your AI models',
              onTap: () => Navigator.pushNamed(context, CustomModelsPage.route),
            ),
            const SizedBox(height: 12),
            const _SectionHeader('Support & Legal'),
            OptionTile(
              icon: Icons.help_outline,
              title: 'Help Center',
              subtitle: 'Get help and support',
              onTap: () => Navigator.pushNamed(context, HelpCenterPage.route),
            ),
            OptionTile(
              icon: Icons.description_outlined,
              title: 'Terms of Service',
              onTap: () => Navigator.pushNamed(context, TermsPage.route),
            ),
            OptionTile(
              icon: Icons.privacy_tip_outlined,
              title: 'Privacy Policy',
              onTap: () =>
                  Navigator.pushNamed(context, PrivacyPolicyPage.route),
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.title);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w700,
          color: AppColors.textPrimary,
        ),
      ),
    );
  }
}
